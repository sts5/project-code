# Use an official Python runtime as a parent image
FROM python:3.7-slim-bullseye
# Set the working directory to /main
WORKDIR /main

# Copy the current directory contents into the container at /app
COPY . /main

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Define environment variable
ENV NAME World

# Expose the port that the application will listen on
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "main.py"]
