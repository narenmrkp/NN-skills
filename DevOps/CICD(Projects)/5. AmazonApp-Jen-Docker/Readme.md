This Project explains how to deploy Amazon Cloned App on EC2 as Docker container through Terraform-Jenkins-Sonar-Docker
In Jenkins:
Install all needed plugins in Manage Jenkins --> Plaugins --> Availability Plugins (Eclipse Temurin, SonarQube Scanner, OWASP, Docker Commons, Docker pipeline, Docker API, docker-build-setup)
Config Java & NodeJs --> Manage Jenkins --> Tools --> Install JDK(17) & NodeJs(16) --> Apply --> Save
In Sonar Dashboard --> Administration --> Security --> Tokens --> Generate Token
Administration–> Configuration–>Webhooks --> create --> Jenkins, <http://jenkins-public-ip:8080>/sonarqube-webhook/ --> Create
Manage Jenkins --> Credentials (Global) --> secret Text --> copy & paste the above Sonar Token as Sonar-token
Manage Jenkins --> Credentials (Global) --> Docker --> username(narian318), password(xxxxxxxx) --> save
Manage Jenkins --> System --> sonar-server, xxxx:9000/, Sonar-token --> Apply & Save
Manage Jenkins --> Tools --> Sonarqube Scanner Installation --> sonar-scanner, Install Automatically(Sonarqube scanner 4.8.0.2856)
Manage Jenkins --> Tools --> DP-Check --> Install Automatically (dependancy check(6.5.1) --> Apply & Save
Manage Jenkins --> Tools --> docker --> Install Automatically(latest)
Jenkins Pipeline --> copy & paste Jenkins pipeline script (from Jenkinsfile) --> Build --> <EC2-public-ip:3000> (To access Amazon App in browser)

terraform destroy --auto-approve

