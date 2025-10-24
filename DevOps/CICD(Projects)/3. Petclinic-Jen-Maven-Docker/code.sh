EC2(t2.large,ubuntu 20.04) with 22,80,443,8080,9000 ports opened
Jenkins, Trivy, sonarqube, docker Install:
  # Jenkins
sudo apt-get update 
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | sudo tee \ 
    /usr/share/keyrings/jenkins-keyring.asc > /dev/null 
echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \ 
    https://pkg.jenkins.io/debian-stable binary/ | sudo tee \ 
    /etc/apt/sources.list.d/jenkins.list > /dev/null 
sudo apt update 
sudo apt install openjdk-17-jdk 
sudo apt install openjdk-17-jre 
sudo systemctl enable jenkins 
sudo systemctl start jenkins 
sudo systemctl status jenkins 
sudo cat  /var/lib/jenkins/secrets/initialAdminPassword    # access publicIP:8080 (Jenkins dashboard, use this password for login)
  # Docker
sudo apt-get update 
sudo apt-get install docker.io -y 
sudo usermod -aG docker $USER 
sudo chmod 777 /var/run/docker.sock  
sudo docker ps 
docker run -d --name sonar -p 9000:9000 sonarqube:lts-community    # access publicIP:9000 (sonarqube, username(admin), password(admin))
  # Trivy
sudo apt-get install wget apt-transport-https gnupg lsb-release -y 
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | gpg --dearmor | sudo tee /usr/share/keyrings/trivy.gpg > /dev/null 
echo "deb [signed-by=/usr/share/keyrings/trivy.gpg] https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list 
sudo apt-get update 
sudo apt-get install trivy -y
---------------------------------------------------------
Install all needed plugins in Manage Jenkins --> Plaugins --> Availability Plugins (Eclipse Temurin, SonarQube Scanner, OWASP, Docker Commons, Docker pipeline, Docker API, docker-build-setup)
Config Java & Maven --> Manage Jenkins --> Tools --> Install JDK & Maven3 --> Apply --> Save
In Sonar Dashboard --> Administration --> Security --> Tokens --> Generate Token
Manage Jenkins --> Credentials (Global) --> secret Text --> copy & paste the above Sonar Token as Sonar-token
Manage Jenkins --> Credentials (Global) --> username(narian318), password(xxxxxxxx) --> save
Manage Jenkins --> System --> sonar-server, xxxx:9000/, Sonar-token --> Apply & Save
Manage Jenkins --> Tools --> Sonarqube Scanner Installation --> sonar-scanner, Install Automatically(Sonarqube scanner 4.8.0.2856)
Manage Jenkins --> Tools --> DP-Check --> Install Automatically (dependancy check(6.5.1) --> Apply & Save
Manage Jenkins --> Tools --> docker --> Install Automatically(latest)

Create a Job (Pipeline) --> Build
pipeline{ 
    agent any 
    tools { 
        jdk 'jdk11' 
        maven 'maven3' 
    } 
    stages{ 
        stage ('clean Workspace'){ 
            steps{ 
                cleanWs() 
            } 
        } 
        stage ('checkout scm') { 
            steps { 
                checkout scmGit(branches: [[name: '*/master']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/Aj7Ay/amazon-eks-jenkins-terraform-aj7.git']]) 
            } 
        } 
        stage ('maven compile') { 
            steps { 
                sh 'mvn clean compile' 
            } 
        } 
        stage ('sonarqube Analysis'){ 
            steps{ 
                script{ 
                    withSonarQubeEnv(credentialsId: 'Sonar-token') { 
                      sh 'mvn sonar:sonar' 
                    } 
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
        stage("OWASP Dependency Check"){ 
            steps{ 
                dependencyCheck additionalArguments: '--scan ./ --format HTML ', odcInstallation: 'DP-Check' 
                dependencyCheckPublisher pattern: '**/dependency-check-report.xml' 
            } 
        } 
        stage ('Build war file'){ 
            steps{ 
                sh 'mvn clean install package' 
            } 
        } 
        stage ('Build and push to docker hub'){ 
            steps{ 
                script{ 
                    withDockerRegistry(credentialsId: 'docker', toolName: 'docker') { 
                        sh "docker build -t petclinic1 ." 
                        sh "docker tag petclinic1 sevenajay/pet-clinic123:latest" 
                        sh "docker push sevenajay/pet-clinic123:latest" 
                   } 
                } 
            } 
        } 
        stage("TRIVY"){ 
            steps{ 
                sh "trivy image sevenajay/pet-clinic123:latest" 
            } 
        } 
        stage ('Deploy to container'){ 
            steps{ 
                sh 'docker run -d --name pet1 -p 8082:8080 sevenajay/pet-clinic123:latest' 
            } 
        } 
    } 
}
-----------------------------------
In EC2 --> docker ps --> check running container --> access our petclinic App through EC2 publicIP:8082 in web browser
