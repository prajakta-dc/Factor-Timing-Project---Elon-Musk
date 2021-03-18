import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate

class Parameters:
    '''
    MACRO variables, no changes needed unless environment changes.
    '''
  
    risk_aversion = 3.0
    risk_premium = 0.1
    trading_days_pa = 252
    rebalance_freq_days = 21
    data_time_step_days = 1

    @staticmethod
    def set_param(risk_aversion = 3.0, 
                  risk_premium = 0.1,
                  trading_days_pa = 252, 
                  rebalance_freq_days = 1,
                  data_time_step_days = 1):
      
        Parameters.risk_aversion = risk_aversion
        Parameters.risk_premium = risk_premium
        Parameters.trading_days_pa = trading_days_pa
        Parameters.rebalance_freq_days =  rebalance_freq_days
        Parameters.data_time_step_days = data_time_step_days



class Simulator:
    def __init__(self, strategy_name,initial_book_size=1):
        '''
        Assume we don't vary simulator's parameters once set
        '''

        self.initial_book_size = initial_book_size
        self.strategy_name = strategy_name

   

    def add_weights_lag(self, weights, lag=1):
        '''
        Allows variation of lags, lag >= 1
        '''
        weights_lag = weights.shift(lag)
        weights_lag.fillna(0, inplace=True)
        return weights_lag
   
    def cal_pnl(self,
                    target_weights,
                    returns,
                    tcost_coeff, 
                    hodling_cost_coeff):
        
            '''
            PnL calculation that allows variation of transaction costs.
            '''

            # Sanity check on the weights and returns matrix
            assert(target_weights.shape == returns.shape)
            assert(sum(returns.index == target_weights.index) == target_weights.shape[0])

            gross_label = '{} gross'.format(self.strategy_name)
            tcost_label = '{} gross - tcost'.format(self.strategy_name)
            tcost_hodl_label = '{} gross - tcost - hcost'.format(self.strategy_name)


            # Data frame that contains the value of the portfolio over time for different simulation settings
            V_t = pd.DataFrame(index=returns.index,
                               columns=[gross_label, tcost_label, tcost_hodl_label])

            idx = target_weights.index

            # Initializing the portfolio value with stated initial book size
            V_t.loc[idx[0], :] = self.initial_book_size

            # Weights of the assets at the start of simulation: Portfolio yet to be formed
            w_start_of_period_gross = [0.0] * len(target_weights.columns)
            w_start_of_period_tc = [0.0] * len(target_weights.columns)
            w_start_of_period_tc_hc = [0.0] * len(target_weights.columns)
            

            # Rebalance condition boolean variable
            rebalance_cond = False

            # Turnover series
            turnover_series_gross = pd.Series(index = idx)
            turnover_series_tc = pd.Series(index = idx)
            turnover_series_tc_hc = pd.Series(index = idx)
            
            # Setting initial value to be = 0 
            turnover_series_gross.loc[idx[0]] = 0
            turnover_series_tc.loc[idx[0]] = 0
            turnover_series_tc_hc.loc[idx[0]] = 0
            
            # Simulation Start Bool
            flag = True

            # Simulation
            for dt in range(1, len(idx)):

                """
                We are working with end of day prices and weights rebalancing.
                Weights are lagged by >= 1 lag.
                We go sequentially from the start of the period with rebalancing to target weights to end of the period
                where value of the portfolio is computed for different simulation settings.

                """
                curr_date = idx[dt]
                prev_date = idx[dt - 1]
                
                step_days = float((curr_date - prev_date).days)
                
                # Given multiplier (below) is used for holding cost, we choose to use ACT/365 convention
                step_multiplier_ann = step_days / 365
                
                curr_target_weights = target_weights.loc[curr_date]
                curr_returns = returns.loc[curr_date]
                
       
                # Period Start:
                # Rebalance condition
                if dt % (Parameters.rebalance_freq_days/Parameters.data_time_step_days) == 0 or flag:
                    rebalance_cond = True
                    flag = False
                else:
                    rebalance_cond = False

                # Rebalancing/Trading the portfolio to the target weights from the prevailing weights if the rebalancing frequency arrives
                if rebalance_cond:
                    
                    # print(curr_target_weights.values ,w_start_of_period_tc)
                    # Transaction cost computation for transaction cost enabled simulation setting
                    # print(curr_target_weights.shape)
                    # print(w_start_of_period_tc.shape)

                    tcost_tc = np.nansum(abs(curr_target_weights.values - w_start_of_period_tc) * tcost_coeff) *  V_t.loc[prev_date, tcost_label]
                    tcost_tc_hc = np.nansum(abs(curr_target_weights.values - w_start_of_period_tc_hc) * tcost_coeff) *  V_t.loc[prev_date, tcost_hodl_label]
        
                    """
                    # Transaction cost computation for transaction cost enabled simulation setting
                    #tcost_tc = abs(curr_risky_weight - w_start_of_period_tc[0]) * V_t.loc[prev_date, tcost_label] * tcost_coeff
                    """
                    
                    """
                    # Registering turnover in the turnover series of respective simulation settings
                    turnover_series_gross.loc[curr_date] = abs(curr_risky_weight - w_start_of_period_gross[0])
                    turnover_series_tc.loc[curr_date] = abs(curr_risky_weight - w_start_of_period_tc[0])
                    """
                    # Registering turnover in the respective turnover series 
                    turnover_series_gross.loc[curr_date] = np.nansum(abs(curr_target_weights - w_start_of_period_gross))
                    turnover_series_tc.loc[curr_date] = np.nansum(abs(curr_target_weights - w_start_of_period_tc))
                    turnover_series_tc_hc.loc[curr_date] = np.nansum(abs(curr_target_weights - w_start_of_period_tc_hc))

                else:

                    # Case of no rebalance
                    tcost_tc = 0
                    tcost_tc_hc = 0
                    turnover_series_gross.loc[curr_date] = 0
                    turnover_series_tc.loc[curr_date] = 0
                    turnover_series_tc_hc.loc[curr_date] = 0

            
                # Simulation setting: transaction costs enabled 
                V_t_minus_1_star_tc = V_t.loc[prev_date, tcost_label] - tcost_tc
                V_t_minus_1_star_tc_hc = V_t.loc[prev_date, tcost_hodl_label] - tcost_tc_hc


                # End of the period
                # Simulation setting: 'gross': No transaction costs

                if rebalance_cond:
                    V_t.loc[curr_date, gross_label] = V_t.loc[prev_date, gross_label] * (1 + np.nansum(curr_target_weights * curr_returns))
                    
                else:
                    V_t.loc[curr_date, gross_label] = V_t.loc[prev_date, gross_label] * (1 + np.nansum(w_start_of_period_gross * curr_returns))     
                                     
                        
                # Simulation setting: Transaction cost enabled
                if rebalance_cond:
                    # Assuming after transaction costs deduction the curren weights are not drastically deviated from the current target weights
                    V_t.loc[curr_date, tcost_label] = V_t_minus_1_star_tc * (1 + np.nansum(curr_target_weights * curr_returns))       
                else:
                    V_t.loc[curr_date, tcost_label] = V_t_minus_1_star_tc * (1 + np.nansum(w_start_of_period_tc * curr_returns))    
                    

                # Simulation setting: Transaction and Holding cost enabled; Holding cost assumed to be deducted at the end of a period
                if rebalance_cond:
                    # Assuming after transaction costs deduction the curren weights are not drastically deviated from the current target weights
                    V_t.loc[curr_date, tcost_hodl_label] = V_t_minus_1_star_tc_hc * (1 + np.nansum(curr_target_weights * curr_returns) - (np.nansum(curr_target_weights * hodling_cost_coeff) * step_multiplier_ann))    
                  
                else:
                    V_t.loc[curr_date, tcost_hodl_label] = V_t_minus_1_star_tc_hc * (1 + np.nansum(w_start_of_period_tc_hc * curr_returns) - (np.nansum(w_start_of_period_tc_hc * hodling_cost_coeff) * step_multiplier_ann))       
                    
                
                
                
                
                # weight of the new positions at the end of the period or beginning of new period (before trading) 
                # Simulation setting: 'gross': No transaction costs
                if rebalance_cond:
                    w_start_of_period_gross =  curr_target_weights * V_t.loc[prev_date, gross_label] * (1 + curr_returns ) / V_t.loc[curr_date, gross_label] 

                else:
                    w_start_of_period_gross =  w_start_of_period_gross * V_t.loc[prev_date, gross_label] * (1 + curr_returns ) / V_t.loc[curr_date, gross_label]
                 

                # Simulation setting: transaction costs enabled
                if rebalance_cond:
                    w_start_of_period_tc = curr_target_weights * V_t_minus_1_star_tc * (1 + curr_returns ) / V_t.loc[curr_date, tcost_label]
                    
                else:
                    w_start_of_period_tc = w_start_of_period_tc * V_t_minus_1_star_tc * (1 + curr_returns ) / V_t.loc[curr_date, tcost_label]
                  
                # Simulation setting: transaction and holding costs enabled
                
                # Approx treatment (Similar to Tcost approx. treatment); assuming the holding cost deduction does not affect the weights drastically
                if rebalance_cond:
                    w_start_of_period_tc_hc = curr_target_weights * V_t_minus_1_star_tc_hc * (1 + curr_returns ) / V_t.loc[curr_date, tcost_hodl_label]
                    
                else:
                    w_start_of_period_tc_hc = w_start_of_period_tc_hc * V_t_minus_1_star_tc_hc * (1 + curr_returns ) / V_t.loc[curr_date, tcost_hodl_label]
                  



            """
            Assumption: We employ log returns on the strategy calculation and using the law of large numbers, later compute the mean of the returns of the strategy; 
            average of log return variables is a unbiased estimator and converges to the true in sample mean as the number of observations grow. For our case, the number of 
            observations are large enough for all practical purposes.
            """
            # Strategy log return
            strat_log_ret = np.log(V_t.astype('float')) - np.log(V_t.astype('float')).shift(1)

            # Putting together the turnover dataframe
            turnover_df = pd.concat([turnover_series_gross, turnover_series_tc, turnover_series_tc_hc],axis = 1)
            turnover_df.columns = [gross_label, tcost_label, tcost_hodl_label]

            return V_t, strat_log_ret, turnover_df
        
class Visualizer:
    def __init__(self):
        pass

    def show_stats(self, stats_data):
        print(" "*60 + "Mean Annualized Performance Statistics")
        print(tabulate(stats_data, headers=stats_data.columns, tablefmt='psql'))

    def plot_pnl(self,
                 pnl_data,
                 strategy_name,
                 initial_book_size,
                 x_label='Time',
                 y_label='Natural Log scale: log($ Amount)',
                 scale = 'log',
                 figsize=[20, 10]):
        title = 'Value of ${} when traded via {}'.format(initial_book_size, strategy_name)
        font = {'family': 'calibri',
                'weight': 'bold',
                'size': 15}

        matplotlib.rc('font', **font)
        fig, axs = plt.subplots(figsize=figsize)

        axs.set_title(title)
        if scale == 'log':
            axs.set_ylabel(y_label)
        elif scale == 'linear':
            axs.set_ylabel('Linear Scale: $ Amount')
        axs.set_xlabel(x_label)

        if isinstance(pnl_data, pd.DataFrame):
            for c in pnl_data.columns:
                if scale == 'log':
                    axs.plot(np.log(pnl_data.loc[:, c].astype(float)), label=c)
                
                elif scale == 'linear':
                    axs.plot(pnl_data.loc[:, c].astype(float), label=c)

        elif isinstance(pnl_data, pd.Series):
            print(pnl_data.name)

            if scale == 'log':
                axs.plot(np.log(pnl_data.astype(float)), label=pnl_data.name)

            elif scale == 'linear':
                axs.plot(pnl_data, label=pnl_data.name)

        plt.grid(True, alpha=0.75)
        plt.legend(loc=0)
        plt.show()
class Indicator:
    def __init__(self):
        pass


    def cal_Sharpe(self, strat_log_ret, returns):
        # Sharpe Computation
        
        # Computing strategy excess returns for different cases: Gross, net of transaction cost & net of tcost and financing cost
        strat_excess_ret = strat_log_ret.copy()
        
        for col in strat_log_ret.columns:
            #Sanity check: Whether the indices match
            assert(sum(strat_log_ret.index == returns.index) == returns.shape[0])
            
            # Converting to arithmetic returns -> excess arithmetic returns -> log excess returns            
            strat_excess_ret[col] = np.log(1 + ((np.exp(strat_log_ret[col]) - 1) - returns['rf_ret']))
        
        
        # Annualized Arithmetic returns/vol based Sharpe ratio
        sharpe = (self.cal_annual_ret(strat_excess_ret, arithmetic = True) 
                  / self.cal_annual_vol(strat_excess_ret, arithmetic = True))


        return sharpe

    def cal_annual_ret(self, strat_log_ret, arithmetic = False):
        # Annualized log returns
        avg_log_ret_pa = strat_log_ret.mean(axis = 0, skipna = True) * Parameters.trading_days_pa / Parameters.data_time_step_days
        
        # Converting Ann log returns to Arithmetic returns
        if arithmetic:
            avg_arithmetic_ret_pa = np.exp(avg_log_ret_pa) - 1
            return avg_arithmetic_ret_pa
            
        return avg_log_ret_pa

    def cal_annual_vol(self, strat_log_ret, arithmetic = False):
        # Annualized volatility of log returns
        avg_log_vol_pa = strat_log_ret.std(axis=0, skipna = True) * np.sqrt(Parameters.trading_days_pa
                                                                            / Parameters.data_time_step_days)
        
        if arithmetic:
            arithmetic_ret = np.exp(strat_log_ret) - 1
            avg_arithmetic_vol_pa = arithmetic_ret.std(axis=0, skipna = True) * np.sqrt(Parameters.trading_days_pa
                                                                                        / Parameters.data_time_step_days)
            return avg_arithmetic_vol_pa
        
        return avg_log_vol_pa
        
    def cal_annual_skew(self, strat_log_ret, arithmetic = False):
        # https://quant.stackexchange.com/a/3678
        
        avg_log_skew_pa = (strat_log_ret.skew(axis=0, skipna = True) 
                           / np.sqrt(Parameters.trading_days_pa / Parameters.data_time_step_days))
        if arithmetic:
            arithmetic_ret = np.exp(strat_log_ret) - 1
            
            avg_arithmetic_skew_pa = (arithmetic_ret.skew(axis=0, skipna = True) 
                                       / np.sqrt(Parameters.trading_days_pa / Parameters.data_time_step_days))
            return avg_arithmetic_skew_pa
        
        return avg_log_skew_pa

    def cal_annual_kurt(self, strat_log_ret, arithmetic = False):
        # https://quant.stackexchange.com/a/3678
        avg_log_kurt_pa = (strat_log_ret.kurt(axis=0, skipna = True) 
                           / (Parameters.trading_days_pa / Parameters.data_time_step_days))
        
        if arithmetic:
            arithmetic_ret = np.exp(strat_log_ret) - 1
            
            avg_arithmetic_kurt_pa =  (arithmetic_ret.kurt(axis=0, skipna = True) 
                                       / (Parameters.trading_days_pa / Parameters.data_time_step_days))
            return avg_arithmetic_kurt_pa
        
        return avg_log_kurt_pa 


    # Max Drawdown

    def cal_mdd(self, V_t):
        # https://quant.stackexchange.com/questions/18094/how-can-i-calculate-the-maximum-drawdown-mdd-in-python
        Roll_Max = V_t.cummax()
        rolling_Drawdown = V_t/Roll_Max - 1.0
        Max_Drawdown = rolling_Drawdown.min()
        return Max_Drawdown

    def cal_turnover(self, turnover_df):
        # Computing two way turnover
        two_turn = turnover_df.mean(axis = 0)
        ann_turnover = two_turn * (Parameters.trading_days_pa / Parameters.data_time_step_days)
        return ann_turnover

    def agg_stats(self, 
                  strat_log_ret,
                  returns, 
                  V_t,
                  turnover_df):

        '''
        Aggregate indicator calculation methods.
        '''
        sharpe = self.cal_Sharpe(strat_log_ret, returns)
        avg_ret_pa = self.cal_annual_ret(strat_log_ret)
        
        # Computing vol, skewness and kurtosis in arithmetic returns space 
        avg_vol_pa = self.cal_annual_vol(strat_log_ret, arithmetic=True)
        avg_skew_pa = self.cal_annual_skew(strat_log_ret, arithmetic=True)
        avg_kurt_pa = self.cal_annual_kurt(strat_log_ret, arithmetic=True)
        
        max_drawdown = self.cal_mdd(V_t)
        mdd = [np.around(x * 100,2) for x in max_drawdown.values]
        mdd = pd.DataFrame(mdd, index = sharpe.index)
       
        ann_turnover = self.cal_turnover(turnover_df) 
        
        
        stats = pd.concat([sharpe, avg_ret_pa*100, avg_vol_pa*100, 
                           ann_turnover * 100, avg_skew_pa, avg_kurt_pa, mdd], 
                          axis=1)
        stats.columns = ['Sharpe Ratio', 'Returns (%)', 'Volatility (%)', 
                         'Turnover (%)','Skewness','Kurtosis','Max Drawdown (%)']
        stats = stats.round(2)
        return stats
