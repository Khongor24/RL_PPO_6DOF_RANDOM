Environment gives reward based on distance
reward -= 0.01 * float(np.sum(np.square(action)))

if distance < 0.05:
    reward += 5.0
elif distance < 0.1:
    reward += 3.0
elif distance < 0.2:
    reward += 1.0
