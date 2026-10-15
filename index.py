from functionCall import predict_result

result = predict_result(
    study_hrs=8,
    attendance=85,
    gender="male"
)

print(result)