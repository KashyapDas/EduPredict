from functionCall import predict_student

print(predict_student(study_hrs=1, attendance=40, gender="male"))   # expect Fail
print(predict_student(study_hrs=1, attendance=95, gender="male"))   # expect Pass
print(predict_student(study_hrs=9, attendance=40, gender="male"))   # expect Pass