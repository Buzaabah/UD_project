def transform_data(data):
  """
  Transforms data by skipping the first row and combining the next two rows into a single row with the format:
  "Second Row\tFirst Row"

  Args:
      data: A list of strings representing the input data.

  Returns:
      A list of strings representing the transformed data.
  """
  transformed_data = []
  index = 1
  while index < len(data):
    row1, row2 = data[index], data[index + 1].strip()  # Strip extra spaces and newlines
    transformed_data.append(f"{row2}\t{row1}")  # Create combined row with tab separator
    index += 2  # Skip two rows at a time
  return transformed_data

# Define file paths
input_file = "Runya-Engli.tsv"
output_file = "Tran.txt"

# Read data from the input file
with open(input_file, "r", encoding="utf-8") as f:
  data = f.readlines()

# Skip the first row
data = data[1:]

# Transform the data
transformed_data = transform_data(data)

# Write transformed data to the output file
with open(output_file, "w", encoding="utf-8") as f:
  f.write("\n".join(transformed_data))

print(f"Data successfully transformed and saved to: {output_file}")





