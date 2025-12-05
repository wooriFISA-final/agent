pipeline {
    agent any
    
    environment {
        DOCKER_USERNAME = 'dongseok0610'
        IMAGE_NAME = 'woorifisa-agent-dev'
        EC2_HOST = '[테스트 EC2 Private IP]'
        BASTION_HOST = '[Bastion Public IP]'
        EC2_USER = 'ubuntu'
        CONTAINER_PORT = '9000'
    }
    
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main_dev', url: 'https://github.com/wooriFISA-final/agent.git'
            }
        }
        
        stage('Docker Login') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh 'echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin'
                }
            }
        }
        
        stage('Build & Push Docker Image') {
            steps {
                script {
                    sh 'docker buildx create --use --name multiplatform || true'
                    sh "docker buildx build --platform linux/amd64 -t ${DOCKER_USERNAME}/${IMAGE_NAME}:latest --push ."
                }
            }
        }
        
        stage('Deploy to Test EC2 via Bastion') {
            steps {
                sshagent(credentials: ['ec2-ssh-key']) {
                    sh """
                        ssh -o StrictHostKeyChecking=no -J ${EC2_USER}@${BASTION_HOST} ${EC2_USER}@${EC2_HOST} '
                            sudo docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:latest
                            sudo docker stop agent-dev || true
                            sudo docker rm agent-dev || true
                            sudo docker run -d \\
                                --name agent-dev \\
                                --restart unless-stopped \\
                                -p ${CONTAINER_PORT}:${CONTAINER_PORT} \\
                                --env-file ~/agent/.env \\
                                ${DOCKER_USERNAME}/${IMAGE_NAME}:latest
                            sudo docker ps
                        '
                    """
                }
            }
        }
    }
    
    post {
        success { 
            echo '✅ Agent (main_dev) 배포 성공!' 
        }
        failure { 
            echo '❌ Agent (main_dev) 배포 실패!' 
        }
    }
}