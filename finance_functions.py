import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


def max_drawdown_levels(series):
    """
    Calculates the maximum drawdown at every point in time for a Pandas Series.

    Parameters:
    series (pd.Series): The input Pandas Series.

    Returns:
    pd.Series: A Pandas Series containing the maximum drawdown at each point in time.

    Explanation:
    1. The `cummax()` method is used to calculate the cumulative maximum of the input Pandas Series. This represents the highest value the series has reached up to each point in time.
    2. The drawdown is calculated as the difference between the current value and the cumulative maximum, divided by the cumulative maximum. This gives the percentage drawdown from the highest value.
    3. The `cummin()` method is used to calculate the maximum drawdown at each point in time, by finding the minimum value of the drawdown series up to that point.

    The resulting Pandas Series contains the maximum drawdown at each point in time, which can be useful for analyzing the risk and performance of a financial time series.

    Example:
    # Example data
    data = pd.Series([100, 105, 110, 115, 120, 115, 110, 105, 100, 95, 90])

    # Calculate the maximum drawdown
    drawdown = max_drawdown(data)

    """
    # Calculate the cumulative maximum
    cummax = series.cummax()

    # Calculate the drawdown
    drawdown = (series - cummax) / cummax

    # Calculate the maximum drawdown
    max_drawdown = drawdown.cummin()

    return max_drawdown







def rolling_std(series, window_size=10, min_periods=1, ddof=0):
    """
    Calculates the moving standard deviation for a Pandas Series.
    
    Parameters:
    series (pd.Series): The input Pandas Series.
    window_size (int): The size of the rolling window (default is 10).
    min_periods (int): The minimum number of non-NA values required to calculate the standard deviation for a window (default is 1).
    ddof (int): The delta degrees of freedom (the divisor used in the calculation is N - ddof, where N represents the number of elements) (default is 0).
    
    Returns:
    pd.Series: A Pandas Series containing the moving standard deviation.
    
    Explanation:
    Here's how the `rolling_std()` function works:
    1. The `rolling()` method is used to create a rolling window over the input Pandas Series.
    2. The `std()` method is used to calculate the standard deviation of the values within each window.
    3. The `window_size` parameter specifies the size of the rolling window (default is 10).
    4. The `min_periods` parameter specifies the minimum number of non-NA values required to calculate the standard deviation for a window (default is 1).
    5. The `ddof` parameter specifies the delta degrees of freedom (the divisor used in the calculation is `N - ddof`, where `N` represents the number of elements) (default is 0).
    The function handles NA values by automatically excluding them from the standard deviation calculation. The output Pandas Series will have NaN values for the first `window_size - 1` elements, as there are not enough data points to calculate the standard deviation.
    You can adjust the `window_size`, `min_periods`, and `ddof` parameters to suit your specific needs.

    Example:
    # Example data
    data = pd.Series([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, None, 60, 65, 70, 75])

    # Calculate the moving standard deviation with a window size of 5
    std_dev = rolling_std(data, window_size=5)
        
    """
    return series.rolling(window=window_size, min_periods=min_periods).std(ddof=ddof)





def point_in_time_correlation(series, backward_window, backward_lag, forward_window, forward_lag):
    """
    Calculates the point-in-time correlation coefficient between the autocorrelation values
    of a backward window and a forward window for a Pandas Series.
    
    Parameters:
    series (pd.Series): The input Pandas Series.
    backward_window (int): The size of the backward window.
    backward_lag (int): The lag for the backward window.
    forward_window (int): The size of the forward window.
    forward_lag (int): The lag for the forward window.
    
    Returns:
    pd.Series: A Pandas Series containing the correlation coefficients for each point in time.
    
    Explanation:
                
    Example:
    # Example data
    data = pd.Series([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, pd.NA, 60, 65, 70, 75, 80])

    # Calculate the point-in-time correlation
    correlation = point_in_time_correlation(data, backward_window=5, backward_lag=2, forward_window=3, forward_lag=1)
   
    """
    # Calculate the backward window autocorrelation
    backward_autocorr = series.shift(backward_lag).rolling(backward_window).corr(series)
    
    # Calculate the forward window autocorrelation
    forward_autocorr = series.shift(-forward_lag).rolling(forward_window).corr(series)
    
    # Calculate the correlation coefficient between the two autocorrelation values
    correlation = pd.Series(index=series.index)
    for i in range(len(series)):
        try:
            correlation.iloc[i] = backward_autocorr.iloc[i].corr(forward_autocorr.iloc[i])
        except (TypeError, ValueError):
            correlation.iloc[i] = pd.NA
    
    return correlation









def max_drawdown_percentage(series):
    """
    Calculates the maximum drawdown at every point in time for a Pandas Series, expressed in percentage terms.
    
    Parameters:
    series (pd.Series): The input Pandas Series.
    
    Returns:
    pd.Series: A Pandas Series containing the maximum drawdown at each point in time, expressed as a percentage.
    
    Example:
    # Example data
    data = pd.Series([100, 105, 110, 115, 120, 115, 110, 105, 100, 95, 90])

    # Calculate the maximum drawdown
    drawdown = max_drawdown(data)

    print(drawdown)
    """
    try:
        # Calculate the cumulative maximum
        cummax = series.cummax()
        
        # Calculate the drawdown
        drawdown = (series - cummax) / cummax * 100
        
        # Calculate the maximum drawdown
        max_drawdown = drawdown.cummin()
        
        return max_drawdown
    
    except (TypeError, ValueError) as e:
        print(f"Error: {e}")
        return pd.Series(dtype=float)




def reverse_max_drawdown_percentage(price_series):
    """
    Calculate the reverse point-in-time maximum drawdown for a Series.
    
    Parameters:
    price_series (pandas.Series): Series containing the prices.
    
    Returns:
    pandas.Series: The reverse point-in-time maximum drawdown.
    
    Explanation:
    The function returns a Series containing the reverse point-in-time maximum drawdown. The reverse max drawdown is calculated as the largest percentage change from a maximum to the previous minimum.
    This implementation uses the built-in `cummax()` and `cummin()` functions of the pandas Series to calculate the cumulative maximum and minimum prices, respectively. Then, the reverse max drawdown is calculated as the percentage change between the cumulative maximum and minimum prices.
        
    Example:    
    # Example Series
    prices = pd.Series([100, 102, 101, 103, 105, 102, 104, 106, 104, 102])
    # Calculate the reverse max drawdown
    reverse_drawdown = reverse_max_drawdown_vectorized(prices)
    # Print the results
    print("Reverse Max Drawdown:")
    print(reverse_drawdown)
        
    """
    try:
        # Check if the input is a Series
        if not isinstance(price_series, pd.Series):
            raise ValueError("Input must be a pandas Series.")
        
        # Calculate the cumulative maximum and minimum prices
        cummax = price_series.cummax()
        cummin = price_series.cummin()
        
        # Calculate the reverse max drawdown
        reverse_drawdown = (cummax - cummin) / cummax * 100
        
        return reverse_drawdown
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def drawdown_elbow_threshold(series, window_size=10, min_periods=1, kernel_bandwidth=0.1):
    """
    Calculates the point-in-time maximum drawdown for a Pandas Series, finds the 'elbow point'
    in the Gaussian Kernel Density Estimate of the drawdown values, and returns the
    corresponding 'threshold' drawdown value in percentage terms.
    
    Parameters:
    series (pd.Series): The input Pandas Series.
    window_size (int): The size of the rolling window for calculating maximum drawdown (default is 10).
    min_periods (int): The minimum number of non-NA values required to calculate the maximum drawdown for a window (default is 1).
    kernel_bandwidth (float): The bandwidth parameter for the Gaussian Kernel Density Estimate (default is 0.1).
    
    Returns:
    float: The 'threshold' maximum drawdown value (in percentage terms) at the elbow point of the Kernel Density Estimate.
    
    Explanation:
    1. The drawdown is calculated as a percentage by multiplying the result by 100.
    2. The `max_drawdown` Series contains the maximum drawdown values in percentage terms.
    3. The `x` array used for the Gaussian Kernel Density Estimate is also in percentage terms.
    
    Example:
    # Example data
    data = pd.Series([100, 105, 110, 115, 120, 115, 110, 105, 100, 95, 90])

    # Calculate the 'threshold' maximum drawdown
    threshold = drawdown_elbow_threshold(data, window_size=5, kernel_bandwidth=0.2)
    print(f"The 'threshold' maximum drawdown is: {threshold:.2f}%")
    """
    try:
        # Calculate the point-in-time maximum drawdown
        cummax = series.cummax()
        drawdown = (series - cummax) / cummax
        max_drawdown = drawdown.rolling(window=window_size, min_periods=min_periods).min() * 100
        
        # Calculate the Gaussian Kernel Density Estimate of the drawdown values
        kde = gaussian_kde(max_drawdown.dropna(), bw_method=kernel_bandwidth)
        
        # Find the 'elbow point' in the KDE
        x = np.linspace(max_drawdown.min(), max_drawdown.max(), 1000)
        y = kde(x)
        peaks, _ = find_peaks(-y, prominence=0.1)
        if len(peaks) > 0:
            elbow_idx = peaks[0]
            elbow_drawdown = x[elbow_idx]
        else:
            elbow_drawdown = max_drawdown.min()
        
        return elbow_drawdown
    
    except (TypeError, ValueError) as e:
        print(f"Error: {e}")
        return pd.NA




def zigzag(df, high_col='high', low_col='low', pct_threshold=3, min_swing=2):
    """
    Calculate the Zigzag indicator on a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the necessary columns (high, low).
    high_col (str): Name of the column containing the high prices.
    low_col (str): Name of the column containing the low prices.
    pct_threshold (float): Percentage threshold for the Zigzag indicator.
    min_swing (int): Minimum number of bars for a swing.
    
    Returns:
    pandas.Series: Previous Zigzag high.
    pandas.Series: Previous Zigzag low.
    pandas.Series: Last point (high or low).
    
    Explanation:
    The `zigzag()` function takes the following parameters:
    - `df`: The input DataFrame containing the `high` and `low` columns.
    - `high_col`: The name of the column containing the high prices (default is 'high').
    - `low_col`: The name of the column containing the low prices (default is 'low').
    - `pct_threshold`: The percentage threshold for the Zigzag indicator (default is 3%).
    - `min_swing`: The minimum number of bars for a swing (default is 2).

    The function returns three Series:
    1. `zigzag_high`: The previous Zigzag high.
    2. `zigzag_low`: The previous Zigzag low.
    3. `last_point`: Whether the last point was a high or a low.

    The function includes exception handling to catch any errors that may occur, such as missing columns in the input DataFrame. If an error occurs, the function will return `None` for all three output Series.
    You can customize the function by adjusting the `pct_threshold` and `min_swing` parameters to suit your needs. The `high_col` and `low_col` parameters can be used to specify the column names if they are different from the default 'high' and 'low'.
    
    Example:
    # Example DataFrame
    df = pd.DataFrame({
        'high': [100, 102, 101, 103, 105, 102, 104, 106, 104, 102],
        'low': [98, 99, 100, 101, 102, 100, 101, 103, 102, 100]
    })

    # Calculate Zigzag
    zigzag_high, zigzag_low, last_point = zigzag(df)

    # Print the results
    print("Previous Zigzag High:")
    print(zigzag_high)
    print("\nPrevious Zigzag Low:")
    print(zigzag_low)
    print("\nLast Point (High or Low):")
    print(last_point)
    """
    try:
        # Check if the required columns exist in the DataFrame
        if high_col not in df.columns or low_col not in df.columns:
            raise ValueError(f"Columns '{high_col}' and '{low_col}' must be present in the DataFrame.")
        
        # Initialize Zigzag Series
        zigzag_high = pd.Series(index=df.index, dtype='float64')
        zigzag_low = pd.Series(index=df.index, dtype='float64')
        last_point = pd.Series(index=df.index, dtype='object')
        
        # Initialize variables for Zigzag calculation
        prev_high = df[high_col][0]
        prev_low = df[low_col][0]
        current_high = prev_high
        current_low = prev_low
        last_high_idx = 0
        last_low_idx = 0
        
        # Calculate Zigzag
        for i in range(1, len(df)):
            high = df[high_col][i]
            low = df[low_col][i]
            
            # Update current high/low
            if high > current_high:
                current_high = high
                last_high_idx = i
            if low < current_low:
                current_low = low
                last_low_idx = i
            
            # Check for new high/low
            if (high - prev_high) / prev_high * 100 >= pct_threshold and i - last_high_idx >= min_swing:
                zigzag_high[i] = current_high
                prev_high = current_high
                current_high = high
                last_high_idx = i
            elif (prev_low - low) / prev_low * 100 >= pct_threshold and i - last_low_idx >= min_swing:
                zigzag_low[i] = current_low
                prev_low = current_low
                current_low = low
                last_low_idx = i
            
            # Determine the last point
            if zigzag_high[i] > 0 and zigzag_low[i] > 0:
                if zigzag_high[i] > zigzag_low[i]:
                    last_point[i] = 'high'
                else:
                    last_point[i] = 'low'
            elif zigzag_high[i] > 0:
                last_point[i] = 'high'
            elif zigzag_low[i] > 0:
                last_point[i] = 'low'
            else:
                last_point[i] = None
        
        return zigzag_high, zigzag_low, last_point
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None





def past_future_window_correlations(input_series, window_sizes=[(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]):
    """
    Processes a Pandas Series and returns a DataFrame with the correlation between the standard deviations
    calculated using different past and future rolling window sizes.
    
    Args:
        input_series (pandas.Series): The input Pandas Series to be processed.
        window_sizes (list of tuples, optional): A list of tuples, where each tuple contains the past and future window sizes to be tested. Defaults to [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)].
        
    Returns:
        pandas.DataFrame: A DataFrame containing the past window size, future window size, and the correlation between the two standard deviation series.
    
    Explanation:
    The function  takes an optional `window_sizes` parameter, which is a list of tuples containing the past and future window sizes to be tested. The function loops through these window sizes, calculates the standard deviations and their correlation, and stores the results in a list. Finally, it converts the list of results to a Pandas DataFrame and returns it.
    The function still includes the error handling to ensure that the input is a Pandas Series. If the input is not a Pandas Series, the function will raise a `TypeError`.

        
    Example:
    # Create a sample Pandas Series
    sample_series = pd.Series([100, 102, 99, 101, 103, 98, 102, 100, 101, 99])

    # Call the function
    result = process_series(sample_series)
    """
    if not isinstance(input_series, pd.Series):
        raise TypeError("Input must be a Pandas Series.")
    
    results = []
    
    for past_window, future_window in window_sizes:
        # Make the series stationary by calculating the percentage change
        pct_change_series = input_series.pct_change()
        
        # Calculate the standard deviation using a past rolling window
        past_std_dev = pct_change_series.rolling(window=past_window).std()
        
        # Calculate the standard deviation using a future rolling window
        future_std_dev = pct_change_series.rolling(window=future_window).std()
        
        # Calculate the correlation between the two standard deviation series
        correlation = past_std_dev.corr(future_std_dev)
        
        results.append({
            "past_window": past_window,
            "future_window": future_window,
            "correlation": correlation
        })
    
    return pd.DataFrame(results)

