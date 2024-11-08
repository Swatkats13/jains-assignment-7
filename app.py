from flask import Flask, render_template, request, session, g
import numpy as np
import matplotlib
import time  # <--- Import time for cache-busting in templates
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key

# Global dictionary to hold data temporarily (for local use; not for production)
app_data = {}

def generate_data(N, mu, beta0, beta1, sigma2, S):
    X = np.random.rand(N)
    noise = mu + np.sqrt(sigma2) * np.random.randn(N)
    Y = beta0 + beta1 * X + noise

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    plot1_path = "static/plot1.png"
    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', linewidth=2, label='Fitted Line')
    plt.title(f"Linear Fit: y = {intercept:.2f} + {slope:.2f}x")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    slopes = []
    intercepts = []

    for _ in range(S):
        X_sim = np.random.rand(N)
        noise_sim = mu + np.sqrt(sigma2) * np.random.randn(N)
        Y_sim = beta0 + beta1 * X_sim + noise_sim

        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    plot2_path = "static/plot2.png"
    plt.figure(figsize=(12, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color='red', label='Slopes')
    plt.hist(intercepts, bins=20, alpha=0.5, color='blue', label='Intercepts')
    plt.axvline(slope, color='red', linestyle='--', linewidth=1, label=f'Observed Slope: {slope:.2f}')
    plt.axvline(intercept, color='blue', linestyle='--', linewidth=1, label=f'Observed Intercept: {intercept:.2f}')
    plt.title("Histogram of Simulated Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot2_path)
    plt.close()

    # Store data in app_data for later use
    app_data['X'] = X
    app_data['Y'] = Y
    app_data['slopes'] = slopes
    app_data['intercepts'] = intercepts
    app_data['slope'] = slope  # Store observed slope
    app_data['intercept'] = intercept  # Store observed intercept

    return X, Y, slope, intercept, plot1_path, plot2_path

def hypothesis_test(X, Y, parameter='slope', null_value=0, test_type='greater', S=1000):
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    observed_slope = model.coef_[0]
    observed_intercept = model.intercept_

    # Use the observed slope or intercept
    observed_value = observed_slope if parameter == 'slope' else observed_intercept
    simulated_values = app_data['slopes'] if parameter == 'slope' else app_data['intercepts']

    # Perform the hypothesis test based on the test type
    if test_type == 'greater':
        p_value = np.mean([sim >= observed_value for sim in simulated_values])
    elif test_type == 'less':
        p_value = np.mean([sim <= observed_value for sim in simulated_values])
    else:  # 'not equal'
        p_value = np.mean([abs(sim - null_value) >= abs(null_value - observed_value) for sim in simulated_values])

    return observed_value, simulated_values, p_value

def plot_hypothesis_test(simulated_values, observed_value, null_value, parameter, test_type, plot_path):
    plt.figure()
    plt.hist(simulated_values, bins=20, alpha=0.5, color='blue', label='Simulated Statistics')
    plt.axvline(observed_value, color='red', linestyle='--', linewidth=2, label=f'Observed {parameter.capitalize()}: {observed_value:.2f}')
    plt.axvline(null_value, color='green', linestyle='-', linewidth=2, label=f'Hypothesized {parameter.capitalize()} (H0): {null_value:.2f}')
    plt.title(f"Hypothesis Test: {parameter.capitalize()} under Null Hypothesis")
    plt.xlabel(parameter.capitalize())
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()


def calculate_confidence_interval(parameter, confidence_level):
    simulated_values = app_data['slopes'] if parameter == 'slope' else app_data['intercepts']
    
    # Verify simulated_values content
    print(f"calculate_confidence_interval - Length of simulated_values: {len(simulated_values)}")
    
    if not simulated_values:
        raise ValueError("No simulated values available for confidence interval calculation.")

    mean_estimate = np.mean(simulated_values)
    standard_error = np.std(simulated_values, ddof=1)

    # Calculate the t-critical value and margin of error
    alpha = 1 - confidence_level / 100
    t_critical = t.ppf(1 - alpha / 2, df=len(simulated_values) - 1)
    margin_of_error = t_critical * standard_error

    # Calculate confidence interval bounds
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error

    # Check if true parameter lies within the confidence interval
    true_value = app_data['slope'] if parameter == 'slope' else app_data['intercept']
    includes_true = ci_lower <= true_value <= ci_upper if true_value is not None else False

    # Log final results before returning
    print(f"calculate_confidence_interval - mean_estimate: {mean_estimate}")
    print(f"calculate_confidence_interval - ci_lower: {ci_lower}")
    print(f"calculate_confidence_interval - ci_upper: {ci_upper}")
    print(f"calculate_confidence_interval - includes_true: {includes_true}")

    return mean_estimate, ci_lower, ci_upper, includes_true


def plot_confidence_interval(simulated_values, mean_estimate, ci_lower, ci_upper, true_value, parameter, confidence_level, plot_path):
    print(f"Saving confidence interval plot at {plot_path}")  # Log before saving plot

    plt.figure()
    plt.scatter(simulated_values, np.zeros_like(simulated_values), color='gray', alpha=0.5, label='Simulated Estimates')
    plt.scatter(mean_estimate, 0, color='blue', label='Mean Estimate')
    plt.hlines(0, ci_lower, ci_upper, color='red', linewidth=2, label=f'{confidence_level}% Confidence Interval')
    plt.axvline(true_value, color='green', linestyle='--', label='True Value')
    plt.title(f'{confidence_level}% Confidence Interval for {parameter.capitalize()}')
    plt.xlabel(f'{parameter.capitalize()} Estimate')
    plt.ylabel('Frequency (as points)')
    plt.legend()
    plt.savefig(plot_path)  # Save the plot to the static folder
    plt.close()
    
    print(f"Confidence interval plot saved at {plot_path}")  # Log after saving



@app.route("/calculate_ci", methods=["POST"])
def calculate_ci():
    if request.method == "POST":
        try:
            # Retrieve parameters and confidence level
            parameter = request.form.get("ci_parameter", "slope")
            confidence_level = float(request.form.get("confidence_level", 95))

            # Select appropriate simulated values and true value based on parameter
            simulated_values = app_data.get("slopes", []) if parameter == "slope" else app_data.get("intercepts", [])
            true_value = session.get("beta1" if parameter == "slope" else "beta0", None)

            # Log the details for debugging
            print(f"calculate_ci - Parameter: {parameter}, Confidence Level: {confidence_level}")
            print(f"calculate_ci - Number of Simulated Values: {len(simulated_values)}")
            print(f"calculate_ci - True Value: {true_value}")

            if not simulated_values:
                print("Error: No simulated values found for confidence interval calculation.")
                return render_template("index.html", error="No simulated values found for CI.")

            # Calculate the confidence interval
            mean_estimate, ci_lower, ci_upper, includes_true = calculate_confidence_interval(parameter, confidence_level)
            print("Confidence interval calculated successfully")

            # Define plot path and plot confidence interval
            plot_path = f"static/ci_plot_{parameter}.png"
            print(f"Preparing to save confidence interval plot at: {plot_path}")

            plot_confidence_interval(simulated_values, mean_estimate, ci_lower, ci_upper, true_value, parameter, confidence_level, plot_path)
            print(f"Confidence interval plot saved at {plot_path}")

            # Render the template with the plot and confidence interval information
            
        
            return render_template(
                "index.html",
                plot1="static/plot1.png",
                plot2="static/plot2.png",
                plot3="static/plot3.png",
                ci_plot=plot_path,
                mean_estimate=mean_estimate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                includes_true=includes_true,
                confidence_level=confidence_level,
                ci_parameter=parameter,
                time=time  # Pass time to the template
            )


        except Exception as e:
            print(f"Error during confidence interval calculation: {e}")
            return render_template("index.html", error=f"Error: {e}")

    return render_template("index.html")






@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        action = request.form.get("action")
        print(f"Received action: {action}")

        if action == "generate_data":
            try:
                # Retrieve and convert form inputs
                N = int(request.form.get("N", 100))
                mu = float(request.form.get("mu", 0.0))
                sigma2 = float(request.form.get("sigma2", 1.0))
                beta0 = float(request.form.get("beta0", 0.0))
                beta1 = float(request.form.get("beta1", 1.0))
                S = int(request.form.get("S", 1000))

                print(f"Generating data with N={N}, mu={mu}, sigma2={sigma2}, beta0={beta0}, beta1={beta1}, S={S}")
                X, Y, slope, intercept, plot1, plot2 = generate_data(N, mu, beta0, beta1, sigma2, S)

                # Store key variables in session for later use
                session['slope'] = slope
                session['intercept'] = intercept
                session['beta0'] = beta0
                session['beta1'] = beta1

                print("Data generation successful.")
                return render_template(
                    "index.html", 
                    plot1=plot1, 
                    plot2=plot2, 
                    N=N, 
                    mu=mu, 
                    sigma2=sigma2, 
                    beta0=beta0, 
                    beta1=beta1, 
                    S=S
                )
            except Exception as e:
                print(f"Error during data generation: {e}")
                return render_template("index.html", error=f"Error during data generation: {e}")

        elif action == "hypothesis_test":
            try:
                if 'X' in app_data and 'Y' in app_data:
                    X = app_data['X']
                    Y = app_data['Y']
                    parameter = request.form.get("test_parameter", "slope")
                    null_value = float(request.form.get("null_value", 0.0))
                    test_type = request.form.get("test_type", "greater")

                    print(f"Running hypothesis test with parameter={parameter}, null_value={null_value}, test_type={test_type}")
                    observed_value, simulated_values, p_value = hypothesis_test(X, Y, parameter, null_value, test_type, S=1000)
                    plot3_path = "static/plot3.png"
                    plot_hypothesis_test(simulated_values, observed_value, null_value, parameter, test_type, plot3_path)

                    print(f"Observed value: {observed_value}, p-value: {p_value}")
                    fun_message = None
                    if p_value <= 0.0001:
                        fun_message = "You've encountered an extremely small p-value (≤ 0.0001), indicating a rare event!"

                    return render_template(
                        "index.html", 
                        plot1="static/plot1.png", 
                        plot2="static/plot2.png", 
                        plot3=plot3_path, 
                        observed_value=observed_value, 
                        p_value=p_value, 
                        fun_message=fun_message,
                        test_parameter=parameter, 
                        null_value=null_value, 
                        test_type=test_type
                    )
            except Exception as e:
                print(f"Error during hypothesis testing: {e}")
                return render_template("index.html", error=f"Error during hypothesis testing: {e}")

    # Render the initial page if no form is submitted
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)
