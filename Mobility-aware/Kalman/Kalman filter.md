<!-- 1404-07-22 -->
<!-- https://www.geeksforgeeks.org/python/kalman-filter-in-python/ -->

### Kalman filter:

- The core problem it solves is: How do you get a **reliable estimate** of an **object's position** when your measurements (e.g., from a GPS or radar) are **noisy** and **unreliable**?
- The Kalman Filter is an optimal recursive algorithm used for estimating the state of a linear dynamic system from a series of noisy measurements.
- It is widely applied in robotics, navigation, finance and any field where accurate tracking and prediction from uncertain data is required.
- The filter effectively fuses observed measurements with prior understanding of the system to provide more precise estimates.

### The Kalman Filter operates in a loop of two main steps:
- Prediction: Estimate the next state using the current estimate (prior knowledge).
- Update: Refine the prediction using the newly observed measurement, optimally balancing model prediction and new data.


### Library:
- https://pypi.org/project/filterpy/
