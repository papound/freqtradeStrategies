import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter
from functools import reduce
import pandas as pd


###########################################################################################################
##                                  BigZ04 by ilya                                                       ##
##                                                                                                       ##
##    https://github.com/i1ya/freqtrade-strategies                                                       ##
##    The stratagy most inspired by iterativ (authors of the CombinedBinHAndClucV6)                      ##
##                                                                                                       ##                                                                                                       ##
###########################################################################################################
##     The main point of this strat is:                                                                  ##
##        -  make drawdown as low as possible                                                            ##
##        -  buy at dip                                                                                  ##
##        -  sell quick as fast as you can (release money for the next buy)                              ##
##        -  soft check if market if rising                                                              ##
##        -  hard check is market if fallen                                                              ##
##        -  11 buy signals                                                                              ##
##        -  stoploss function preventing from big fall                                                  ##
##        -  no sell signal. Whether ROI or stoploss =)                                                  ##
##                                                                                                       ##
###########################################################################################################
##                 GENERAL RECOMMENDATIONS                                                               ##
##                                                                                                       ##
##   For optimal performance, suggested to use between 2 and 4 open trades, with unlimited stake.        ##
##                                                                                                       ##
##   As a pairlist you can use VolumePairlist.                                                           ##
##                                                                                                       ##
##   Ensure that you don't override any variables in your config.json. Especially                        ##
##   the timeframe (must be 5m).                                                                         ##
##                                                                                                       ##
##   exit_profit_only:                                                                                   ##
##       True - risk more (gives you higher profit and higher Drawdown)                                  ##
##       False (default) - risk less (gives you less ~10-15% profit and much lower Drawdown)             ##
##                                                                                                       ##
###########################################################################################################
##               DONATIONS 2 @iterativ (author of the original strategy)                                 ##
##                                                                                                       ##
##   Absolutely not required. However, will be accepted as a token of appreciation.                      ##
##                                                                                                       ##
##   BTC: bc1qvflsvddkmxh7eqhc4jyu5z5k6xcw3ay8jl49sk                                                     ##
##   ETH: 0x83D3cFb8001BDC5d2211cBeBB8cB3461E5f7Ec91                                                     ##
##                                                                                                       ##
###########################################################################################################


class InkkerSlower(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.028,         # I feel lucky!
        "10": 0.018,
        "40": 0.005,
        "180": 0.018,        # We're going up?
    }


    stoploss = -0.99 # effectively disabled.

    timeframe = '5m'
    inf_1h = '1h'

    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_entry_signal = False

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'force_entry': 'market',
        'force_exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    buy_params = {
        #############
        # Enable/Disable conditions
        "buy_condition_0_enable": True,
    }

    sell_params = {
        #############
        # Enable/Disable conditions
        "sell_condition_0_enable": True,
    }

    ############################################################################

    # Buy

    # buy_condition_0_enable = CategoricalParameter([True, False], default=True, space='entry', optimize=False, load=True)

    #Sell
    # sell_condition_0_enable = CategoricalParameter([True, False], default=True, space='exit', optimize=False, load=True)

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:

        return True


    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        return False


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Manage losing trades and open room for better ones.

        if (current_profit > 0):
            return 0.99
        else:
            trade_time_50 = trade.open_date_utc + timedelta(minutes=50)

            # Trade open more then 60 minutes. For this strategy it's means -> loss
            # Let's try to minimize the loss

            if (current_time > trade_time_50):

                try:
                    number_of_candle_shift = int((current_time - trade_time_50).total_seconds() / 300)
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    candle = dataframe.iloc[-number_of_candle_shift].squeeze()

                    # We are at bottom. Wait...
                    if candle['rsi_1h'] < 30:
                        return 0.99

                    # Are we still sinking? 
                    if candle['close'] > candle['ema_200']:
                        if current_rate * 1.025 < candle['open']:
                            return 0.01 

                    if current_rate * 1.015 < candle['open']:
                        return 0.01

                except IndexError as error:

                    # Whoops, set stoploss at 10%
                    return 0.1

        return 0.99

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        informative_1h['ema_25'] = ta.EMA(informative_1h, timeperiod=25)
        informative_1h['ema_10'] = ta.EMA(informative_1h, timeperiod=10)
        informative_1h['ema_5'] = ta.EMA(informative_1h, timeperiod=5)
        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # Bollinger Band
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # informative_1h['bb_lowerband'] = bollinger['lower']
        # informative_1h['bb_middleband'] = bollinger['mid']
        # informative_1h['bb_upperband'] = bollinger['upper']

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # dataframe['bb_lowerband'] = bollinger['lower']
        # dataframe['bb_middleband'] = bollinger['mid']
        # dataframe['bb_upperband'] = bollinger['upper']

        # dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=48).mean()

        # EMA
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_75'] = ta.EMA(dataframe, timeperiod=75)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_25'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)

        #IFTCOMBO
        ccilength = 5
        wmalength = 9
        dataframe['cci'] = ta.CCI(dataframe['high'], dataframe['low'], dataframe['close'], window=ccilength, constant=0.015, fillna=False)
        dataframe['v11'] = (dataframe['cci'].divide(4)).multiply(0.1)
        dataframe['v21'] = ta.WMA(dataframe['v11'], window=wmalength)
        dataframe['result1'] = np.exp(dataframe['v21'].multiply(2))
        dataframe['iftcombo'] = (dataframe['result1'].subtract(1)).divide(dataframe['result1'].add(1))

        # with pd.option_context('display.max_rows', 30,
        #                'display.max_columns', None,
        #                'display.precision', 3,
        #                ):
        #     print(dataframe['iftcombo'])

        #CRSI
        # CRSI (3, 2, 100)
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi'] =  (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(dataframe['close'], 100)) / 3

        # MACD 
        dataframe['macd'], dataframe['signal'], dataframe['hist'] = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # SMA
        # dataframe['sma_5'] = ta.EMA(dataframe, timeperiod=5)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    # def nz(self, d) -> int:
    #     if d is None:
    #         return 0
    #     else:
    #         return d

    # def updown(s: DataFrame) -> float:
    #     isEqual = s['close'] == s['close'].shift(1)
    #     isGrowing = s['close'] > s['close'].shift(1)
    #     ud = 0.0
    #     if s > s[1]:
    #         if nz(ud[1]) >= 0:
    #             nz(ud[1]) + 1 
    #         else:
    #             ud = 1
    #     else:
    #         if s < s[1]:
    #             if nz(ud[1]) <= 0:
    #                 ud = nz(ud[1]) - 1 
    #             else:
    #                 ud = -1
    #         else:
    #             ud = 0
    #     return ud

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    # RSI Calculation
    def get_rsi(close, lookback):
        scr = close.diff()
        up = []
        down = []
        for i in range(len(scr)):
            if scr[i] < 0:
                up.append(0)
                down.append(scr[i])
            else:
                up.append(scr[i])
                down.append(0)
        up_series = pd.Series(up)
        down_series = pd.Series(down).abs()
        up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
        down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
        rs = up_ewm/down_ewm
        rsi = 100-(100/(1+rs))
        rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
        rsi_df = rsi_df.dropna()
        return rsi_df[3:]

        # ibm['rsi_14'] = get_rsi(ibm['close'], 14)
        # ibm = ibm.dropna() 
        # print(ibm)

    # trading strategy
    def implement_rsi_strategy(prices, rsi):
        buy_price = []
        sell_price = []
        rsi_signal = []
        signal = 0

        for i in range(len(rsi)):
            if rsi[i-1] > 30 and rsi[i] < 30:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    rsi_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    rsi_signal.append(0)
            elif rsi[i-1] < 70 and rsi[i] > 70:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    rsi_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    rsi_signal.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)

        return buy_price, sell_price, rsi_signal

        # buy_price, sell_price, rsi_signal = implement_rsi_strategy(ibm['close'], ibm['rsi_14'])

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(
            (
                # self.buy_condition_0_enable.value &
                # ปิดเหนือ EMA200
                (dataframe['close'] > dataframe['ema_200']) &
                #EMA5 > EMA10
                (dataframe['ema_5'] > dataframe['ema_10']) &
                #EMA5 > EMA25
                (dataframe['ema_5'] > dataframe['ema_25']) &
                #IFTCOMBO < -0.6
                (dataframe['iftcombo'] < -0.6) &
                #MACD histogram ขาขึ้น
                (dataframe['hist'] > 0) &
                (dataframe['hist'].shift(2) < 0) &
                #CRSI < 35
                (dataframe['crsi'] < 35) &
                #Volume > 0
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []

        # dataframe.loc[
        #     (
        #         (dataframe['close'] > dataframe['bb_middleband'] * 1.01) &                  # Don't be gready, sell fast
        #         (dataframe['volume'] > 0) # Make sure Volume is not 0
        #     )
        #     ,
        #     'sell'
        # ] = 0

        conditions.append(
            (
                # ปิดเหนือ EMA200
                (dataframe['close'] < dataframe['ema_200']) &
                #EMA5 > EMA10
                (dataframe['ema_5'] < dataframe['ema_10']) &
                (dataframe['ema_5'].shift(2) > dataframe['ema_10'].shift(2)) &
                #EMA5 > EMA25
                (dataframe['ema_5'] < dataframe['ema_25']) &
                (dataframe['ema_5'].shift(2) > dataframe['ema_25'].shift(2)) &
                #IFTCOMBO > 0.6
                (dataframe['iftcombo'] > 0.6) &
                #MACD histogram ขาลง
                (dataframe['hist'] < 0) &
                (dataframe['hist'].shift(2) > 0) &
                #CRSI > 64.5
                (dataframe['crsi'] > 64.5) &
                #Volume > 0
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1
        return dataframe