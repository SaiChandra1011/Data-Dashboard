import streamlit as st
import pandas as pd
import numpy as np # Import NumPy
import plotly.express as px
import io

# --- Page Configuration (Apply Theming Here) ---
st.set_page_config(
    layout="wide",
    page_title="Advanced Data Dashboard",
    page_icon="📊", # Add a page icon
    initial_sidebar_state="expanded", # Keep sidebar open initially
    menu_items={
        'Get Help': 'https://www.streamlit.io',
        'Report a bug': None,
        'About': "# This is an *Advanced* Data Dashboard!"
    }
)

# Define custom themes (Optional - Streamlit has good defaults)
# You can customize these colors further if needed.
# By default, Streamlit uses 'auto' theme detection based on OS settings.

# --- Helper Function ---
# Function to convert df to csv for download
def to_csv(df):
    """Converts a Pandas DataFrame to a CSV string for download."""
    output = io.StringIO()
    df.to_csv(output, index=False, encoding='utf-8')
    return output.getvalue()

# --- Main App Logic ---
st.title("📊 Advanced Data Dashboard")
st.markdown("Upload your CSV data and explore it using the options below.")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully!")
    try:
        # Read CSV - consider adding na_values if needed, e.g., na_values=['None', 'N/A', '']
        df = pd.read_csv(uploaded_file)

        # --- Data Preprocessing (FIXED DATETIME CONVERSION) ---
        date_keywords = ['date', 'time'] # Keywords to identify potential date columns

        for col in df.columns:
            # Attempt to convert object columns to datetime ONLY if column name suggests it
            # Also check if the column doesn't consist mostly of numeric strings
            if df[col].dtype == 'object':
                # Basic check if column name implies date/time
                is_potential_date = any(keyword in col.lower() for keyword in date_keywords)

                if is_potential_date:
                    # Further check: attempt conversion only if values look like dates
                    # This avoids converting columns like 'Update Time (sec)' if they are numeric strings
                    try:
                        # Sample non-NA values to check format before converting whole column
                        sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                        if sample_val:
                             # Attempt parsing the sample - if it works, likely a date column
                             pd.to_datetime(sample_val)
                             # Apply conversion to the whole column
                             df[col] = pd.to_datetime(df[col], errors='coerce') # Coerce errors to NaT
                             st.write(f"Attempted datetime conversion for column: {col}") # Debug message
                    except (ValueError, TypeError, OverflowError, AttributeError):
                         # If sample parsing fails, likely not a standard date format we handle easily
                         st.write(f"Column '{col}' named like date, but sample value '{sample_val}' not parsed as date. Kept as object.")
                         pass # Keep as object
                    except IndexError:
                        # Column might be empty or all NA
                        pass # Keep as object
                # else: column name doesn't suggest date, leave as object.

            # Ensure successfully converted columns are standardized (might be redundant)
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                 df[col] = pd.to_datetime(df[col])

        st.subheader("Data Preview")
        st.dataframe(df.head()) # Display the dataframe *after* preprocessing

        st.subheader("Data Summary")
        # Use an expander for the potentially large summary table
        with st.expander("Show Data Summary Statistics", expanded=False):
             # Handle potential error if describe fails (e.g., on mixed types after coerce)
             try:
                 st.write(df.describe(include='all'))
             except Exception as desc_e:
                 st.error(f"Could not generate data summary: {desc_e}")
                 st.write("Preview of data types:")
                 st.write(df.dtypes)


        # --- Filtering ---
        st.sidebar.header("Filtering Options")

        # Store filters in session state
        if 'filters' not in st.session_state:
            st.session_state.filters = {}

        all_columns = df.columns.tolist()
        filter_column = st.sidebar.selectbox(
            "Select column to add/update filter",
            ["None"] + all_columns,
            key="filter_col_select",
            help="Choose a column to apply a filter. Select 'None' to clear selection."
        )

        if filter_column != "None":
            # Ensure the column exists before proceeding (safety check)
            if filter_column in df.columns:
                col_type = df[filter_column].dtype
                # Check for empty column to avoid errors on nunique() or min()/max()
                if df[filter_column].isna().all():
                    st.sidebar.warning(f"Column '{filter_column}' is empty or all null values. Cannot filter.")
                else:
                    unique_vals_count = df[filter_column].nunique()

                    # Categorical Filter (Object or few unique numerics/dates treated as categorical)
                    # Check if dtype is object OR if nunique is small (adjust threshold as needed)
                    if pd.api.types.is_object_dtype(col_type) or \
                       (pd.api.types.is_numeric_dtype(col_type) and unique_vals_count < 15 and unique_vals_count > 0) or \
                       (pd.api.types.is_datetime64_any_dtype(col_type) and unique_vals_count < 15 and unique_vals_count > 0):

                        try:
                            unique_values = ["All"] + sorted(df[filter_column].dropna().unique().tolist())
                            # Use session state to remember the last selection for this filter
                            default_filter_val = "All"
                            if filter_column in st.session_state.filters and st.session_state.filters[filter_column][0] == 'eq':
                                # Check if the saved filter value still exists in the current unique values
                                if st.session_state.filters[filter_column][1] in unique_values:
                                    default_filter_val = st.session_state.filters[filter_column][1]
                                else:
                                    # Saved value is stale, reset to 'All' and remove filter
                                    if filter_column in st.session_state.filters:
                                        del st.session_state.filters[filter_column]

                            selected_value = st.sidebar.selectbox(
                                f"Filter by {filter_column}",
                                unique_values,
                                index=unique_values.index(default_filter_val) if default_filter_val in unique_values else 0,
                                key=f"filter_{filter_column}"
                            )
                            if selected_value != "All":
                                st.session_state.filters[filter_column] = ('eq', selected_value)
                            elif filter_column in st.session_state.filters and st.session_state.filters[filter_column][0] == 'eq':
                                 del st.session_state.filters[filter_column]
                        except TypeError as te:
                             st.sidebar.error(f"Error creating filter for {filter_column}: Mixed data types? {te}")


                    # Numerical Filter
                    elif pd.api.types.is_numeric_dtype(col_type):
                        try:
                            min_val, max_val = float(df[filter_column].min()), float(df[filter_column].max())
                            # Use session state to remember the last selection
                            default_range = (min_val, max_val)
                            if filter_column in st.session_state.filters and st.session_state.filters[filter_column][0] == 'range':
                                 # Ensure saved range is within current min/max
                                 saved_min, saved_max = st.session_state.filters[filter_column][1]
                                 default_range = (max(min_val, saved_min), min(max_val, saved_max))


                            selected_range = st.sidebar.slider(
                                f"Filter by {filter_column}",
                                min_val, max_val, default_range,
                                key=f"filter_{filter_column}"
                            )
                            if selected_range != (min_val, max_val):
                                 st.session_state.filters[filter_column] = ('range', selected_range)
                            elif filter_column in st.session_state.filters and st.session_state.filters[filter_column][0] == 'range':
                                 del st.session_state.filters[filter_column]
                        except Exception as e:
                            st.sidebar.error(f"Could not create slider for {filter_column}: {e}")

                    # Date Filter
                    elif pd.api.types.is_datetime64_any_dtype(col_type):
                         try:
                             # Drop NaT values before getting min/max
                             valid_dates = df[filter_column].dropna()
                             if not valid_dates.empty:
                                 min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
                                 # Use session state to remember the last selection
                                 default_date_range = [min_date, max_date]
                                 if filter_column in st.session_state.filters and st.session_state.filters[filter_column][0] == 'date_range':
                                     # Ensure stored dates are date objects and within current range
                                     stored_range = st.session_state.filters[filter_column][1]
                                     try:
                                         saved_start = pd.to_datetime(stored_range[0]).date()
                                         saved_end = pd.to_datetime(stored_range[1]).date()
                                         default_date_range = [max(min_date, saved_start), min(max_date, saved_end)]
                                     except Exception:
                                         # Handle case where saved range is invalid
                                         if filter_column in st.session_state.filters:
                                             del st.session_state.filters[filter_column]


                                 selected_date_range = st.sidebar.date_input(
                                     f"Filter by {filter_column}",
                                     value=default_date_range,
                                     min_value=min_date,
                                     max_value=max_date,
                                     key=f"filter_{filter_column}"
                                  )
                                 if len(selected_date_range) == 2:
                                     start_date_selected, end_date_selected = selected_date_range
                                     # Check if selected range is different from the full available range
                                     if (start_date_selected != min_date) or (end_date_selected != max_date):
                                        st.session_state.filters[filter_column] = ('date_range', selected_date_range)
                                     # If it's the full range, remove the filter if it exists
                                     elif filter_column in st.session_state.filters and st.session_state.filters[filter_column][0] == 'date_range':
                                        del st.session_state.filters[filter_column]
                                 # Handle case where user might clear the date input (len != 2)
                                 elif filter_column in st.session_state.filters and st.session_state.filters[filter_column][0] == 'date_range':
                                     del st.session_state.filters[filter_column]
                             else:
                                 st.sidebar.warning(f"Column '{filter_column}' contains no valid dates to filter.")
                         except Exception as e:
                             st.sidebar.error(f"Could not process date filter for {filter_column}: {e}")

                    else:
                        st.sidebar.warning(f"Filtering not implemented for column '{filter_column}' with type: {col_type}")
            else:
                 st.sidebar.error(f"Selected filter column '{filter_column}' not found in DataFrame.")


        # Apply filters
        filtered_df = df.copy()
        active_filters_str = []
        # Make a copy of filter items to avoid modifying dict during iteration if errors occur
        current_filters = list(st.session_state.filters.items())

        if current_filters:
            for col, (ftype, val) in current_filters:
                 # Ensure column still exists before trying to filter
                if col not in filtered_df.columns:
                    st.warning(f"Filter column '{col}' no longer exists. Removing filter.")
                    if col in st.session_state.filters:
                        del st.session_state.filters[col]
                    continue # Skip to next filter

                try:
                    if ftype == 'eq':
                        # Handle potential NaN comparison issues if filter value is NaN
                        if pd.isna(val):
                             filtered_df = filtered_df[filtered_df[col].isna()]
                        else:
                             filtered_df = filtered_df[filtered_df[col] == val]
                        active_filters_str.append(f"`{col}` == `{val}`")
                    elif ftype == 'range':
                        # Ensure column is numeric before range comparison
                        if pd.api.types.is_numeric_dtype(filtered_df[col]):
                             filtered_df = filtered_df[(filtered_df[col] >= val[0]) & (filtered_df[col] <= val[1])]
                             active_filters_str.append(f"`{val[0]}` ≤ `{col}` ≤ `{val[1]}`")
                        else:
                             st.warning(f"Cannot apply numeric range filter: Column '{col}' is not numeric. Removing filter.")
                             if col in st.session_state.filters: del st.session_state.filters[col]
                             st.experimental_rerun()
                    elif ftype == 'date_range':
                         # Ensure comparison is between datetime objects
                         start_date = pd.to_datetime(val[0]).normalize() # Normalize to remove time part for date comparison
                         end_date = pd.to_datetime(val[1]).normalize() # Normalize end date
                         # Ensure the column is also datetime before comparison
                         if pd.api.types.is_datetime64_any_dtype(filtered_df[col]):
                             # Normalize column for comparison if it contains time info
                             col_dates_normalized = filtered_df[col].dt.normalize()
                             filtered_df = filtered_df[(col_dates_normalized >= start_date) & (col_dates_normalized <= end_date)]
                             active_filters_str.append(f"`{val[0]}` ≤ `{col}` ≤ `{val[1]}`")
                         else:
                             st.warning(f"Cannot apply date range filter: Column '{col}' is not datetime type. Removing filter.")
                             if col in st.session_state.filters: del st.session_state.filters[col]
                             st.experimental_rerun()

                except Exception as e:
                    st.error(f"Error applying filter on '{col}': {e}. Removing filter.")
                    # Remove problematic filter and rerun
                    if col in st.session_state.filters:
                        del st.session_state.filters[col]
                    st.experimental_rerun()


        st.subheader("Filtered Data")
        if active_filters_str:
            st.write("Active Filters: " + " AND ".join(active_filters_str))
        else:
            st.write("No filters applied.")

        if filtered_df.empty:
            st.warning("Filtered data is empty based on current selections.")
        else:
            st.dataframe(filtered_df)
            # --- Download Button ---
            try:
                csv_data = to_csv(filtered_df)
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv_data,
                    file_name='filtered_data.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Could not prepare data for download: {e}")

        # --- Aggregation ---
        st.sidebar.header("Aggregation Options")
        # Use columns for better layout in sidebar
        col1_agg, col2_agg = st.sidebar.columns(2)

        with col1_agg:
            group_by_cols = st.multiselect("Group by", all_columns, key="group_cols")
            agg_funcs = st.multiselect("Functions", ['sum', 'mean', 'median', 'count', 'min', 'max', 'std'], key="agg_funcs")

        with col2_agg:
             # Get numeric columns from the *filtered* dataframe
            numeric_cols_agg = filtered_df.select_dtypes(include=np.number).columns.tolist()
            agg_cols = st.multiselect("Aggregate", numeric_cols_agg, key="agg_cols")


        if group_by_cols and agg_cols and agg_funcs:
            # Check if selected aggregation columns still exist and are numeric
            valid_agg_cols = [col for col in agg_cols if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col])]
            invalid_agg_cols = [col for col in agg_cols if col not in valid_agg_cols]

            if invalid_agg_cols:
                st.sidebar.warning(f"Columns selected for aggregation are no longer numeric or available: {', '.join(invalid_agg_cols)}. Please re-select.")
            elif valid_agg_cols: # Proceed only if there are valid columns
                st.subheader("Aggregated Data")
                try:
                    agg_dict = {col: agg_funcs for col in valid_agg_cols}
                    # Use observed=True for potentially better performance with categorical group keys
                    aggregated_df = filtered_df.groupby(group_by_cols, as_index=False, observed=True).agg(agg_dict)

                    # Flatten MultiIndex columns if created
                    if isinstance(aggregated_df.columns, pd.MultiIndex):
                        aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]

                    st.dataframe(aggregated_df)
                except Exception as e:
                    st.error(f"Error during aggregation: {e}")
            else:
                st.sidebar.warning("No valid numeric columns selected for aggregation.")

        # Provide feedback if aggregation cannot run
        elif st.sidebar.button("Run Aggregation", key="run_agg_btn"):
             if not group_by_cols: st.sidebar.warning("Select 'Group by' column(s).")
             if not agg_cols: st.sidebar.warning("Select column(s) to 'Aggregate'.")
             if not agg_funcs: st.sidebar.warning("Select aggregation 'Function(s)'.")


        # --- Plotting ---
        st.sidebar.header("Plotting Options")
        plot_types = ["Bar", "Line", "Scatter", "Histogram", "Box"]
        plot_type = st.sidebar.selectbox("Select plot type", plot_types, key="plot_type")

        st.subheader(f"{plot_type} Plot")

        plot_generated = False
        fig = None

        # Ensure filtered_df is not empty before attempting plots
        if not filtered_df.empty:
            try:
                # Use columns for plot options
                col1_plot, col2_plot = st.sidebar.columns(2)
                # Get columns from the *filtered* dataframe for selection
                plot_all_columns = filtered_df.columns.tolist()


                if plot_type in ["Bar", "Line", "Scatter"]:
                    with col1_plot:
                        # Set sensible default index if columns exist
                        x_idx = 0 if not plot_all_columns else min(1, len(plot_all_columns)-1)
                        x_col = st.selectbox("X-axis", plot_all_columns, key="plot_x", index=x_idx)
                    with col2_plot:
                        y_idx = 0 if not plot_all_columns else min(2, len(plot_all_columns)-1)
                        y_col = st.selectbox("Y-axis", plot_all_columns, key="plot_y", index=y_idx)
                    color_col = st.sidebar.selectbox("Color (Optional)", ["None"] + plot_all_columns, key="plot_color")
                    color_arg = None if color_col == "None" else color_col

                    if x_col and y_col:
                        df_to_plot = filtered_df.copy()
                        title = f"{plot_type} Plot: {y_col} vs {x_col}" + (f" colored by {color_arg}" if color_arg else "")

                        # Ensure selected columns exist in the dataframe to plot
                        if x_col in df_to_plot.columns and y_col in df_to_plot.columns and (color_arg is None or color_arg in df_to_plot.columns):
                            if plot_type == "Line":
                                # Sort data by x-axis for line plots if it's numeric or datetime
                                if pd.api.types.is_numeric_dtype(df_to_plot[x_col]) or pd.api.types.is_datetime64_any_dtype(df_to_plot[x_col]):
                                     df_to_plot = df_to_plot.sort_values(by=x_col)
                                fig = px.line(df_to_plot, x=x_col, y=y_col, color=color_arg, title=title, markers=True) # Added markers
                            elif plot_type == "Bar":
                                fig = px.bar(df_to_plot, x=x_col, y=y_col, color=color_arg, title=title)
                            elif plot_type == "Scatter":
                                fig = px.scatter(df_to_plot, x=x_col, y=y_col, color=color_arg, title=title)
                            plot_generated = True
                        else:
                            st.warning("Selected columns for plotting are not available in the filtered data.")


                elif plot_type in ["Histogram", "Box"]:
                    with col1_plot:
                         # Suggest numeric columns first for these plot types from filtered data
                         numeric_cols_plot = filtered_df.select_dtypes(include=np.number).columns.tolist()
                         other_cols_plot = filtered_df.select_dtypes(exclude=np.number).columns.tolist()
                         plot_val_options = numeric_cols_plot + other_cols_plot
                         if not plot_val_options: # Handle case with no columns
                              st.warning("No columns available for plotting in the filtered data.")
                              val_col = None
                         else:
                              val_col = st.selectbox(f"Value Column", plot_val_options, key="plot_val")
                    with col2_plot:
                         color_col = st.selectbox("Group/Color (Optional)", ["None"] + plot_all_columns, key="plot_color_histbox")
                    color_arg = None if color_col == "None" else color_col

                    if val_col: # Proceed only if a value column is selected
                        df_to_plot = filtered_df.copy()
                        title_suffix = f" of {val_col}" + (f" grouped by {color_arg}" if color_arg else "")

                        # Ensure selected columns exist
                        if val_col in df_to_plot.columns and (color_arg is None or color_arg in df_to_plot.columns):
                            if plot_type == "Histogram":
                                if pd.api.types.is_numeric_dtype(df_to_plot[val_col]):
                                     fig = px.histogram(df_to_plot, x=val_col, color=color_arg, title=f"Histogram{title_suffix}")
                                     plot_generated = True
                                else:
                                     st.warning(f"Histogram requires a numeric 'Value Column'. '{val_col}' is not numeric.")
                            elif plot_type == "Box":
                                # Box plot: Y is typically numeric, X is categorical (optional)
                                y_arg = val_col if pd.api.types.is_numeric_dtype(df_to_plot[val_col]) else None
                                x_arg = color_arg # Use the grouping column for X if provided

                                if y_arg: # If the main value column is numeric, use it for Y
                                     fig = px.box(df_to_plot, x=x_arg, y=y_arg, color=x_arg, # Color by the x-grouping if exists
                                                  title=f"Box Plot{title_suffix}")
                                     plot_generated = True
                                elif x_arg: # If main value isn't numeric, but grouping is provided
                                     st.warning(f"Value column '{val_col}' is not numeric. Showing distribution based on grouping column '{x_arg}'.")
                                     try:
                                         # For categorical box plot, often don't need Y explicitly
                                         fig = px.box(df_to_plot, x=x_arg, color=x_arg, title=f"Box Plot grouped by {x_arg}")
                                         plot_generated = True
                                     except Exception as box_e:
                                         st.error(f"Could not generate Box plot for categorical grouping '{x_arg}': {box_e}")
                                else: # No numeric Y and no grouping X
                                     st.error("Box plot requires a numeric 'Value Column' or a 'Group/Color' column.")
                        else:
                             st.warning("Selected columns for plotting are not available in the filtered data.")


                # Display the plot or info message
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_generated:
                    # A warning/error should have been shown above if fig is None but plot_generated is True
                    pass
                # else: # Removed redundant else condition
                #    st.info("Select columns in the sidebar to generate the plot.")

            except Exception as e:
                st.error(f"An error occurred during plotting: {e}")
                st.exception(e) # Show traceback for debugging
        else:
            # Handle case where filtered_df is empty before plotting section
            st.info("No data available to plot based on current filters.")


    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty or could not be parsed.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.exception(e)

# Message when no file is uploaded
else:
    st.info("👈 Upload a CSV file using the sidebar to begin analysis.")

# Clean up session state (filters) if the file is removed or never uploaded
# Consider clearing filters if the uploaded file changes name/type
# if 'current_filename' not in st.session_state or st.session_state.current_filename != uploaded_file.name:
#     if 'filters' in st.session_state: del st.session_state.filters
#     st.session_state.current_filename = uploaded_file.name if uploaded_file else None

if uploaded_file is None and 'filters' in st.session_state:
    # Decide whether to clear filters when no file is present
    # del st.session_state.filters # Uncomment to clear
    pass # Keep filters for potential re-upload
