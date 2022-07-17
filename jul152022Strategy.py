# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from asyncio import base_tasks
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce
import math

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class Jul152022Strategy(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '5m'
    inf_1h = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space='sell', optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)

    PeriodF = 13
    PeriodS = 55
    EnableSmooth = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    #Strategy

    '''
    Long  / Buy Condition
    1.    Green Line  Cross Up   White Line  (LFS)
    2.    Green candle
    3.    Firefly    Histogram  Value  above middle > 50   , Histogram Show Green
    4.    HARSI  RSI Over Lay  Above or Cross up Middle Line     , RSI Over Lay  Value = > 30
    Stop Loss   = Swing High or White Line
    Take Profit
    TP 1  = 30%  RR 1.5-3 /  TP2  50%
    1.     RSI Over Lay (Yellow Line)   > 38
    2.    Firefly    Histogram  Value  below middle > 50   , Histogram Show Green
    3.    Green Candle

    Take Profit & Close Position
    1.            Green Line  Cross Down   White Line  (LFS)
    '''

    def na(self, val):
        return val != val

    def nz(self, x, y=None):
        if isinstance(x, np.generic):
            return x.fillna(y or 0)
        if x != x:
            if y is not None:
                return y
            return 0
        return x

    def barssince(self, condition, occurrence=0):
        cond_len = len(condition)
        occ = 0
        since = 0
        res = float('nan')
        while cond_len - (since+1) >= 0:
            cond = condition[cond_len-(since+1)]
            if cond and not cond != cond:
                if occ == occurrence:
                    res = since
                    break
                occ += 1
            since += 1
        return res

    def valuewhen(self, condition, source, occurrence=0):
        res = float('nan')
        since = self.barssince(condition, occurrence)
        if since is not None:
            res = source[-(since+1)]
        return res

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return [("ETH/USDT", "5m"), ("BTC/USDT", "15m"),]
    
    def calc_zlema(self, s: DataFrame, l) -> DataFrame:
        ema1=ta.EMA(s, l)
        ema2=ta.EMA(ema1, l)
        d=ema1-ema2
        return ema1+d

    def ma(self, s: DataFrame, l, bt: bool) -> DataFrame:
        d=np.where(np.array(bt) == False,ta.EMA(s,l),self.calc_zlema(s,l))
        return d

    def calulateFireflyIndicator(self, dataframe: DataFrame, metadata: dict)-> DataFrame:
        m=10 #Lookback Length
        n1=3 #Signal Smoothing
        a_s=False #Double smooth Osc
        dataframe['v2']=(dataframe['high']+dataframe['low']+dataframe['close']*2)/4
        dataframe['v3']=(self.ma(dataframe['v2'].fillna(0.0), n1, False))
        dataframe['v4']=dataframe['v2'].fillna(0.0).std(axis=0, skipna=True)
        dataframe['v5']=(dataframe['v2'].fillna(0.0)-dataframe['v3'].fillna(0.0))*100/np.where(dataframe['v4']==0,1,dataframe['v4'].fillna(0.0))
        dataframe['v6']=self.ma(dataframe['v5'].fillna(0.0),n1, False)
        dataframe['v7']=np.where(a_s, self.ma(dataframe['v6'].fillna(0.0), n1, False), dataframe['v6'].fillna(0.0))
        dataframe['ww']=(self.ma(dataframe['v7'],m,False)+100)/2-4
        dataframe['mm']=ta.MAX(dataframe['ww'].fillna(0.0), timeperiod=n1)
        dataframe['wwmm_min']=np.minimum(dataframe['ww'].fillna(0.0), dataframe['mm'].fillna(0.0))
        dataframe['wwmm_max']=np.maximum(dataframe['ww'].fillna(0.0),dataframe['mm'].fillna(0.0))
        dataframe['d']=np.where(dataframe['ww'].fillna(0.0)>50, dataframe['wwmm_min'].fillna(0.0), np.where(dataframe['mm'].fillna(0.0)< 50, dataframe['wwmm_max'].fillna(0.0), None))
        dataframe['dc']= np.where(dataframe['d'].fillna(0.0)>50, np.where(dataframe['d'].fillna(0.0)>dataframe['d'].fillna(0.0).shift(1), "green","orange"), np.where(dataframe['d'].fillna(0.0)<dataframe['d'].fillna(0.0).shift(1), "red", "orange"))
        dataframe['fireflyHistogramValue']= dataframe['d']
        dataframe['fireflyHistogramColor']= dataframe['dc']
        return dataframe
    
    def blackcat_AMA(self, Period, dataframe: DataFrame) -> DataFrame:
        # Vars:
        Fastest = 0.6667
        Slowest = 0.0645
        AMA = 0.00
        dataframe['Price']=dataframe['hl2']
        Price = dataframe['Price']
        dataframe['Diff'] = np.absolute(Price - Price.shift(1).fillna(0.0))
        dataframe['Index']=pd.DataFrame(list(range(len(Price.index))))
        dataframe['Signal'] = np.absolute(Price - Price.shift(Period).fillna(0.0))
        dataframe['Noise'] = np.add(dataframe['Diff'], Period)
        dataframe['efRatio'] = dataframe['Signal'] / dataframe['Noise']
        dataframe['Smooth'] = np.power(dataframe['efRatio'] * (Fastest - Slowest) + Slowest, 2)
        dataframe['AdaptMA'] = pd.DataFrame(0.0, index=range(len(Price.index)), columns=range(1), dtype=np.float64)
        dataframe['AMA'] = np.where(dataframe['Index']<= Period, Price, dataframe['AdaptMA'].shift(1).fillna(0.0) + dataframe['Smooth'] * (Price - dataframe['AdaptMA'].fillna(0.0).shift(1).fillna(0.0)))
        return dataframe['AMA']
    
    def calculateBlackCatIndicator(self, dataframe: DataFrame, metadata: dict)-> DataFrame:
        PeriodF = 13
        PeriodS = 55
        EnableSmooth = False
        dataframe['hl2']=(dataframe['high']+dataframe['low'])/2
        dataframe['hl2'] = dataframe['hl2'].fillna(0.0)
        if EnableSmooth:
            dataframe['AMAValF'] =  ta.LINEARREG(self.blackcat_AMA(PeriodF, dataframe, PeriodF, 0))
            dataframe['AMAValS'] = ta.LINEARREG(self.blackcat_AMA(PeriodS, dataframe, PeriodS, 0))
        else: 
            dataframe['AMAValF'] =  self.blackcat_AMA(PeriodF, dataframe)
            dataframe['AMAValS'] = self.blackcat_AMA(PeriodS, dataframe)
        dataframe['BlackCat_Long']=np.where(qtpylib.crossed_above(dataframe['AMAValF'], dataframe['AMAValS']),"Long", "N/A")
        dataframe['BlackCat_Short']=np.where(qtpylib.crossed_below(dataframe['AMAValF'], dataframe['AMAValS']),"Short", "N/A")
        return dataframe
    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # assert self.dp, "DataProvider is required for multiple timeframes."
        # # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # # EMA
        # informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        # informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # informative_1h['ema_25'] = ta.EMA(informative_1h, timeperiod=25)
        # informative_1h['ema_10'] = ta.EMA(informative_1h, timeperiod=10)
        # informative_1h['ema_5'] = ta.EMA(informative_1h, timeperiod=5)
        # # RSI
        # informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # Bollinger Band
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # informative_1h['bb_lowerband'] = bollinger['lower']
        # informative_1h['bb_middleband'] = bollinger['mid']
        # informative_1h['bb_upperband'] = bollinger['upper']

        return informative_1h

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # The indicators for the 1h informative timeframe
        # informative_1h = self.informative_1h_indicators(dataframe, metadata)
        # dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        # dataframe = self.normal_tf_indicators(dataframe, metadata)

        # Momentum Indicators
        # ------------------------------------
        dataframe = self.calulateFireflyIndicator(dataframe, metadata)
        dataframe = self.calculateBlackCatIndicator(dataframe, metadata)

        print("Orig Dataframe")
        with pd.option_context('display.max_rows', 30,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
            print(dataframe['fireflyHistogramColor'])

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # # Plus Directional Indicator / Movement
        # dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        # # Minus Directional Indicator / Movement
        # dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # # Aroon, Aroon Oscillator
        # aroon = ta.AROON(dataframe)
        # dataframe['aroonup'] = aroon['aroonup']
        # dataframe['aroondown'] = aroon['aroondown']
        # dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        # # Awesome Oscillator
        # dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # # Keltner Channel
        # keltner = qtpylib.keltner_channel(dataframe)
        # dataframe["kc_upperband"] = keltner["upper"]
        # dataframe["kc_lowerband"] = keltner["lower"]
        # dataframe["kc_middleband"] = keltner["mid"]
        # dataframe["kc_percent"] = (
        #     (dataframe["close"] - dataframe["kc_lowerband"]) /
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        # )
        # dataframe["kc_width"] = (
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        # )

        # # Ultimate Oscillator
        # dataframe['uo'] = ta.ULTOSC(dataframe)

        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        # dataframe['cci'] = ta.CCI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # # Stochastic Slow
        # stoch = ta.STOCH(dataframe)
        # dataframe['slowd'] = stoch['slowd']
        # dataframe['slowk'] = stoch['slowk']

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # # Stochastic RSI
        # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
        # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
        # stoch_rsi = ta.STOCHRSI(dataframe)
        # dataframe['fastd_rsi'] = stoch_rsi['fastd']
        # dataframe['fastk_rsi'] = stoch_rsi['fastk']

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # # ROC
        # dataframe['roc'] = ta.ROC(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        # Bollinger Bands - Weighted (EMA based instead of SMA)
        # weighted_bollinger = qtpylib.weighted_bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=20, stds=2
        # )
        # dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        # dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        # dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        # dataframe["wbb_percent"] = (
        #     (dataframe["close"] - dataframe["wbb_lowerband"]) /
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        # )
        # dataframe["wbb_width"] = (
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) /
        #     dataframe["wbb_middleband"]
        # )

        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # # SMA - Simple Moving Average
        # dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        # dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        # dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        # dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        # dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        # dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)

        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Hammer: values [0, 100]
        # dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        # dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        # dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        # dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Morningstar: values [0, 100]
        # dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        # dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        # dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        # dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        # dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        # dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        # dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        # dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        # dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        # dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        # dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        # dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        # dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        # dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        '''
        Long  / Buy Condition
        1.    Green Line  Cross Up   White Line  >>LFS    /   Yellow Line (AMAValF)  Cross Up  Purple Line (AMAValS)   >> Blackcat
        2.    Green candle
        3.    Firefly    Histogram  Value  above middle > 50   , Histogram Show Green
        4.    HARSI  RSI Over Lay  Above Median Line (Default Value = 0 )   / Bar Green / RSI Over Lay  Value > 20 (OB = Over Bought : Default Value = 20)
        5.    IFTCOMBO > 0.60
        Stop Loss   = Swing Low 
        Take Profit
        TP 1  = 30%  RR 1.5-3 /  TP2  50%
        1.     Green Candle
        2.    HARSI  : Bar green / RSI Over Lay (Yellow Line)   > 35 (OB Extreme = Overbought Extreme : Default Value = 30)
        3.    Firefly    Histogram  Value  Above middle > 50   , Histogram Show Green
        4.    IFTCOMBO > 0.60

        Take Profit & Close Position
        Green Line  Cross Down   White Line  (LFS)  /      Yellow Line (AMAValF)  Cross Down  Purple Line (AMAValS)   >> Blackcat
        '''
        conditionsLong = []
        conditionsShort = []
        
        conditionsLong.append(
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value)) &
                (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard: tema below BB middle
                (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard: tema is raising
                (dataframe['rsi'] > 30) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            )
            # 'enter_long'] = 1
        )

        if conditionsLong:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditionsLong),
                'enter_long'
            ] = 1


        conditionsShort.append(
                (
                    # Signal: RSI crosses above 70
                    (qtpylib.crossed_above(dataframe['rsi'], self.short_rsi.value)) &
                    (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard: tema above BB middle
                    (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard: tema is falling
                    (dataframe['rsi'] < 30) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
        )
            
        if conditionsShort:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditionsShort),
                'enter_short'
            ] = 1
            # 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        '''
        Short  / Sell Condition
        1.    Green Line  Cross Up   White Line  >>LFS    /   Yellow Line (AMAValF)  Cross Down Purple Line (AMAValS)   >> Blackcat
        2.    Red Candle
        3.    Firefly  >>  Histogram  Value  Below middle < 50   , Histogram Show Red
        4.    HARSI  RSI Over Lay  Below Median Line (Default Value = 0 )  / Bar RED / RSI Over Lay  Value  <  -20 (OS = Over Sell : Default Value = -20)
        6.    IFTCOMBO <  -0.60
        Stop Loss   = Swing H
        Take Profit
        TP 1  = 30%  RR 1.5-3 /  TP2  50%
        1.    Red Candle
        2.    HARSI : Bar Red / RSI Over Lay (Yellow Line)   < 35 (OS Extreme = OverSell Extreme : Default Value =  -30)
        3.    Firefly  :  Histogram  Value  below middle < 50   , Histogram Show Red
        4.    IFTCOMBO <  -0.60

        Take Profit & Close Position
        Green Line  Cross Up   White Line  (LFS)  /   /   Yellow Line (AMAValF)  Cross Up  Purple Line (AMAValS)   >> Blackcat
        '''
        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &
                (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard: tema above BB middle
                (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard: tema is falling
                (dataframe['rsi'] > 70) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),

            'exit_long'] = 1

        dataframe.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(dataframe['rsi'], self.exit_short_rsi.value)) &
                # Guard: tema below BB middle
                (dataframe['tema'] <= dataframe['bb_middleband']) &
                (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard: tema is raising
                (dataframe['rsi'] < 70) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1

        return dataframe
