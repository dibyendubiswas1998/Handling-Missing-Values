import handle_missing_values.missing_value as mp

# Drop Na Values:
data_path = "Data/missing.csv"
before_, after_, data = mp.drop_na(data_path=data_path)
print(before_)
print(after_)


# Forward & Backward Fill:
before_, after_, data = mp.forward_backward_fill(data_path=data_path, cols=['Age', 'Course', 'Salary'], method='ffill')
print(before_)
print(after_)
print(data)


# Mean, Median & Mode:
before_, after_, data = mp.Mean_Median_Mode(data_path=data_path, cols=['Age', 'Salary'], method='mean')
print(before_)
print(after_)
print(data)


# Arbitary Value:
before_, after_, data = mp.Arbitary_Value(data_path=data_path, col='Age', val=34)
# before_, after_, data = mp.Arbitary_Value(data_path=data_path, col='Course', val='C1')
# before_, after_, data = mp.Arbitary_Value(data_path=data_path, col='Salary', val=37000)
print(before_)
print(after_)
print(data)


# Random Sample Imputation:
before_, after_, data = mp.Random_Sample_imputation(data_path=data_path, cols=['Age', 'Course', 'Salary'])
print(before_)
print(after_)
print(data)


# Removing Rows & Columns:
# Rows
before_, after_, data = mp.Removing_rows_cols(data_path=data_path, rows_num=[1, 2, 3])
print(before_)
print(after_)
print(data)
# # Columns:
before_, after_, data = mp.Removing_rows_cols(data_path=data_path, cols=['Salary'])
print(before_)
print(after_)
print(data)


# Apply KNN Imputer:
before_, after_, data = mp.Apply_KNN_Imputer(data_path=data_path, cols=['Age', 'Salary'])
print(before_)
print(after_)
print(data)


# Third Standard deviation:
before_, after_, data = mp.Third_std(data_path=data_path, cols=['Age', 'Salary'])
print(before_)
print(after_)
print(data)