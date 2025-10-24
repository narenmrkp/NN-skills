This Project explains how to deploy Amazon Cloned App on EC2 as Docker container through Terraform-Jenkins-Sonar-Docker
In Jenkins:
Install all needed plugins in Manage Jenkins --> Plaugins --> Availability Plugins (Eclipse Temurin, SonarQube Scanner, OWASP, Docker Commons, Docker pipeline, Docker API, docker-build-setup)
Config Java & NodeJs --> Manage Jenkins --> Tools --> Install JDK(17) & NodeJs(16) --> Apply --> Save
In Sonar Dashboard --> Administration --> Security --> Tokens --> Generate Token
Administration–> Configuration–>Webhooks --> create --> Jenkins, <http://jenkins-public-ip:8080>/sonarqube-webhook/ --> Create
Manage Jenkins --> Credentials (Global) --> secret Text --> copy & paste the above Sonar Token as Sonar-token
Manage Jenkins --> Credentials (Global) --> username(narian318), password(xxxxxxxx) --> save
Manage Jenkins --> System --> sonar-server, xxxx:9000/, Sonar-token --> Apply & Save
Manage Jenkins --> Tools --> Sonarqube Scanner Installation --> sonar-scanner, Install Automatically(Sonarqube scanner 4.8.0.2856)
Manage Jenkins --> Tools --> DP-Check --> Install Automatically (dependancy check(6.5.1) --> Apply & Save
Manage Jenkins --> Tools --> docker --> Install Automatically(latest)
Jenkins Pipeline --> copy & paste Jenkins pipeline script --> Build --> <EC2-public-ip:3000> (To access Amazon App in browser)

terraform destroy --auto-approve
# Jenkins Pipeline
pipeline{
    agent any
    tools{
        jdk 'jdk17'
        nodejs 'node16'
    }
    environment {
        SCANNER_HOME=tool 'sonar-scanner'
    }
    stages {
        stage('clean workspace'){
            steps{
                cleanWs()
            }
        }
        stage('Checkout from Git'){
            steps{
                git branch: 'main', url: 'https://github.com/Aj7Ay/Amazon-FE.git'
            }
        }
        stage("Sonarqube Analysis "){
            steps{
                withSonarQubeEnv('sonar-server') {
                    sh ''' $SCANNER_HOME/bin/sonar-scanner -Dsonar.projectName=Amazon \
                    -Dsonar.projectKey=Amazon '''
                }
            }
        }
        stage("quality gate"){
           steps {
                script {
                    waitForQualityGate abortPipeline: false, credentialsId: 'Sonar-token'
                }
            }
        }
        stage('Install Dependencies') {
            steps {
                sh "npm install"
            }
        }
    }
}
stage('OWASP FS SCAN') {
            steps {
                dependencyCheck additionalArguments: '--scan ./ --disableYarnAudit --disableNodeAudit', odcInstallation: 'DP-Check'
                dependencyCheckPublisher pattern: '**/dependency-check-report.xml'
            }
        }
        stage('TRIVY FS SCAN') {
            steps {
                sh "trivy fs . > trivyfs.txt"
            }
        }
stage("Docker Build & Push"){
            steps{
                script{
                   withDockerRegistry(credentialsId: 'docker', toolName: 'docker'){
                       sh "docker build -t amazon ."
                       sh "docker tag amazon sevenajay/amazon:latest "
                       sh "docker push sevenajay/amazon:latest "
                    }
                }
            }
        }
        stage("TRIVY"){
            steps{
                sh "trivy image sevenajay/amazon:latest > trivyimage.txt"
            }
        }
stage('Deploy to container'){
            steps{
                sh 'docker run -d --name amazon -p 3000:3000 sevenajay/amazon:latest'
            }
        }
