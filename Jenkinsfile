pipeline {
    agent any
    stages {
        stage('Build Docker Image') {
            steps {
                // Build the Docker image
                bat 'docker build -t my-flask-app .'
            }
        }
        stage('Run Container') {
            steps {
                // Run the Docker container and capture the container ID
                script {
                    env.CONTAINER_ID = bat(script: 'docker run -d -p 5000:5000 my-flask-app', returnStdout: true).trim()
                }
            }
            post {
                always {
                    // Stop and remove the container using the captured container ID
                    bat "docker stop ${env.CONTAINER_ID}"
                    bat "docker rm ${env.CONTAINER_ID}"
                }
            }
        }
        stage('Test Application') {
            steps {
                // Test the application using curl to check if it responds on localhost
                bat 'curl http://localhost:5000'
            }
        }
    }
}

