# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from asyncio import base_tasks
import imp
from h11 import Data
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from typing import Optional


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
    timeframe = '1h'
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

    PeriodF = 10
    PeriodS = 55
    EnableSmooth = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

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

    max_leverage = 25.0
    #Set Leverage
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        if side=='long':
            return 1.0
        else:
            return 50.0

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

    def ma(self, s: DataFrame, l, bt) -> DataFrame:
        d=np.where(bt == False,ta.EMA(s,l),self.calc_zlema(s,l))
        return d

    def calulateFireflyIndicator(self, dataframe: DataFrame, metadata: dict)-> DataFrame:
        m=10 #Lookback Length
        n1=3 #Signal Smoothing
        a_s=False #Double smooth Osc
        dataframe['bt'] = pd.DataFrame(False, index=range(len(dataframe.index)), columns=range(1), dtype=bool)
        dataframe['v2']=((dataframe['high']+dataframe['low']+dataframe['close']*2)/4)
        dataframe['v2']=dataframe['v2'].fillna(0.0)
        dataframe['v3']=self.ma(dataframe['v2'], m, dataframe['bt'])
        dataframe['v3']=dataframe['v3'].fillna(0.0)
        dataframe['v4']=ta.STDDEV(dataframe['v2'],timeperiod=m)
        dataframe['v4']=dataframe['v4'].fillna(0.0)
        dataframe['v5']=(dataframe['v2']-dataframe['v3'])*100/np.where(dataframe['v4']==0.0,1.0,dataframe['v4'])
        dataframe['v5']=dataframe['v5'].fillna(0.0)
        dataframe['v6']=self.ma(dataframe['v5'],n1, dataframe['bt'])
        dataframe['v6']=dataframe['v6'].fillna(0.0)
        dataframe['v7']=np.where(a_s, self.ma(dataframe['v6'], n1, dataframe['bt']), dataframe['v6'])
        dataframe['ww']=(self.ma(dataframe['v7'],m,dataframe['bt'])+100)/2-4
        dataframe['ww'] = dataframe['ww'].fillna(0.0)
        dataframe['mm']=ta.MAX(dataframe['ww'], timeperiod=n1)
        dataframe['mm']=dataframe['mm'].fillna(0.0)
        dataframe['wwmm_min']=np.minimum(dataframe['ww'], dataframe['mm'])
        dataframe['wwmm_max']=np.maximum(dataframe['ww'],dataframe['mm'])
        dataframe['d']=np.where(dataframe['ww']>50, dataframe['wwmm_min'], np.where(dataframe['mm']< 50, dataframe['wwmm_max'], 0.0))
        dataframe['dc']= np.where(dataframe['d']>50, np.where(dataframe['d']>dataframe['d'].shift(1), 1.0,0.0), np.where(dataframe['d']<dataframe['d'].shift(1), -1.0, 0.0))
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
        dataframe['BlackCat_Status']=np.where(qtpylib.crossed_above(dataframe['AMAValF'], dataframe['AMAValS']),"Long", np.where(qtpylib.crossed_below(dataframe['AMAValF'], dataframe['AMAValS']),"Short", "N/A"))
        return dataframe

    def calculateIFTComboIndicator(self, dataframe: DataFrame, metadata: dict)-> DataFrame:
        #IFTCOMBO
        ccilength = 5
        wmalength = 9
        dataframe['cci'] = ta.CCI(dataframe['high'], dataframe['low'], dataframe['close'], window=ccilength, constant=0.015, fillna=False)
        dataframe['v11'] = (dataframe['cci']/4)*0.1
        dataframe['v21'] = ta.WMA(dataframe['v11'], window=wmalength)
        dataframe['result1'] = np.exp(dataframe['v21']*2)
        dataframe['iftcombo'] = (dataframe['result1']-1)/(dataframe['result1']+1)
        dataframe['iftcombo'] = dataframe['iftcombo'].fillna(0.0)
        return dataframe

    #HARSI
    def f_zrsi(self, dataframe: DataFrame, length)-> DataFrame:
        d=ta.RSI(dataframe, timeperiod=length)
        return d

    def f_rsi(self, _source: DataFrame, _length, _mode )-> DataFrame:
        _source['_zrsi'] = self.f_zrsi( _source, _length)
        _source['_smoothed']=np.where(np.isnan(_source['_smoothed'].shift(1)), _source['_zrsi'], (_source['_smoothed'].shift(1)+_source['_zrsi'])/2)
        if _mode :
            return _source['_smoothed']
        else:
            return _source['_zrsi']

    #RSI Heikin-Ashi generation function
    def f_rsiHeikinAshi(self, dataframe: DataFrame, _length ) -> DataFrame:
        i_smoothing = 5
        dataframe['_closeRSI'] = self.f_zrsi(dataframe['ha_close'], _length)
        dataframe['_openRSI'] = np.where(np.isnan(dataframe['_closeRSI'].shift(1)), dataframe['_closeRSI'], dataframe['_closeRSI'].shift(1))
        dataframe['_highRSI_raw'] = self.f_zrsi(dataframe['ha_high'], _length)
        dataframe['_lowRSI_raw'] = self.f_zrsi(dataframe['ha_low'], _length)
        dataframe['_highRSI']=np.maximum(dataframe['_highRSI_raw'],dataframe['_lowRSI_raw'])
        dataframe['_lowRSI']=np.minimum(dataframe['_highRSI_raw'],dataframe['_lowRSI_raw'])
        dataframe['_close'] = ( dataframe['_openRSI'] + dataframe['_highRSI'] + dataframe['_lowRSI'] + dataframe['_closeRSI'] ) / 4
        dataframe['_open'] = pd.DataFrame(None, index=range(len(dataframe.index)), columns=range(1), dtype=np.float64)
        dataframe['_open'] = np.where(np.isnan(dataframe['_open'].shift(i_smoothing)),(dataframe['_openRSI'] + dataframe['_closeRSI'])/2, ( ( dataframe['_open'].shift(1)*i_smoothing ) + dataframe['_close'].shift(1) ) / (i_smoothing+1))
        dataframe['_high']= np.maximum(dataframe['_highRSI'], np.maximum( dataframe['_open'], dataframe['_close']))
        dataframe['_low']= np.minimum( dataframe['_lowRSI'], np.minimum( dataframe['_open'], dataframe['_close']))
        dataframe['f_rsiHeikinAshi_bar_color']=np.where(dataframe['_close'] > dataframe['_open'], 1.0, -1.0)
        return dataframe

    def calculateHARSIIndicator(self, dataframe: DataFrame, metadata: dict)-> DataFrame:
        i_lenHARSI = 14
        dataframe = self.f_rsiHeikinAshi(dataframe, i_lenHARSI)
        return dataframe

    def calculateLFSIndicator(self, dataframe: DataFrame, metadata: dict)-> DataFrame:
        longlen=100
        keylen=80
        shortlen=50
        # longema = ta.ema(src, longlen)
        # shortema = ta.ema(src, shortlen)
        # keyema= ta.ema(src,keylen)
        dataframe['long_ema'] = ta.EMA(dataframe, timeperiod=longlen)
        dataframe['short_ema'] = ta.EMA(dataframe, timeperiod=shortlen)
        dataframe['key_ema'] = ta.EMA(dataframe, timeperiod=keylen)
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

        # # Chart type
        # # ------------------------------------
        # Heikin Ashi Strategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Momentum Indicators
        # ------------------------------------
        dataframe = self.calulateFireflyIndicator(dataframe, metadata)
        dataframe = self.calculateBlackCatIndicator(dataframe, metadata)
        dataframe = self.calculateIFTComboIndicator(dataframe, metadata)
        dataframe = self.calculateHARSIIndicator(dataframe, metadata)
        dataframe = self.calculateLFSIndicator(dataframe, metadata)

        # print("Orig Dataframe")
        # with pd.option_context('display.max_rows', 30,
        #                'display.max_columns', None,
        #                'display.precision', 3,
        #                ):
        #     print(dataframe)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

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

        Short  / Sell Condition
        1.    Green Line  Cross Up   White Line  >>LFS    /   Yellow Line (AMAValF)  Cross Down Purple Line (AMAValS)   >> Blackcat
        2.    Red Candle
        3.    Firefly  >>  Histogram  Value  Below middle < 50   , Histogram Show Red
        4.    HARSI  RSI Over Lay  Below Median Line (Default Value = 0 )  / Bar RED / RSI Over Lay  Value  <  -20 (OS = Over Sell : Default Value = -20)
        6.    IFTCOMBO <  -0.60
        Stop Loss   = Swing H
        '''
        conditionsLong = []
        conditionsShort = []
        
        conditionsLong.append(
            (
                #1
                (qtpylib.crossed_above(dataframe['short_ema'], dataframe['key_ema']) | qtpylib.crossed_above(dataframe['AMAValF'], dataframe['AMAValS'])) &
                #2
                (dataframe['open'] < dataframe['close']) &
                #3
                (dataframe['fireflyHistogramColor'] == 1.0) &
                (dataframe['fireflyHistogramValue'] > 50) &
                #4
                (dataframe['_closeRSI'] > 20) &
                (dataframe['f_rsiHeikinAshi_bar_color'] == 1.0) &
                #5
                (dataframe['iftcombo'] > 0.6) &
                #etc
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            )
            # 'enter_long'] = 1
        )

        if conditionsLong:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditionsLong),
                ['enter_long', 'enter_tag']
            ] = (1, 'enter_long')


        conditionsShort.append(
                (
                    #1
                    (qtpylib.crossed_below(dataframe['short_ema'], dataframe['key_ema']) | qtpylib.crossed_below(dataframe['AMAValF'], dataframe['AMAValS'])) &
                    #2
                    (dataframe['open'] > dataframe['close']) &
                    #3
                    (dataframe['fireflyHistogramColor'] == -1.0) &
                    (dataframe['fireflyHistogramValue'] < 50) &
                    #4
                    (dataframe['_closeRSI'] < -20) &
                    (dataframe['f_rsiHeikinAshi_bar_color'] == -1.0) &
                    #5
                    (dataframe['iftcombo'] < -0.6) &
                    #etc
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
        )
            
        if conditionsShort:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditionsShort),
                ['enter_short', 'enter_tag']
            ] = (1, 'enter_short')
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
        Take Profit - Long
        TP 1  = 30%  RR 1.5-3 /  TP2  50%
        1.     Green Candle
        2.    HARSI  : Bar green / RSI Over Lay (Yellow Line)   > 35 (OB Extreme = Overbought Extreme : Default Value = 30)
        3.    Firefly    Histogram  Value  Above middle > 50   , Histogram Show Green
        4.    IFTCOMBO > 0.60

        Take Profit & Close Position
        Green Line  Cross Down   White Line  (LFS)  /      Yellow Line (AMAValF)  Cross Down  Purple Line (AMAValS)   >> Blackcat
        
        Take Profit - Short
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
                #1
                (qtpylib.crossed_below(dataframe['short_ema'], dataframe['key_ema']) | qtpylib.crossed_below(dataframe['AMAValF'], dataframe['AMAValS'])) &
                #etc
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),

            ['exit_long', 'exit_tag']] = (1, 'exit_long')

        dataframe.loc[
            (
                #1
                (qtpylib.crossed_above(dataframe['short_ema'], dataframe['key_ema']) | qtpylib.crossed_above(dataframe['AMAValF'], dataframe['AMAValS'])) &
                #etc
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
           ['exit_short', 'exit_tag']] = (1, 'exit_short')

        return dataframe
