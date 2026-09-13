// Mirrors .github/workflows/ci.yml (Python test suite + TypeScript validator build/test),
// each stage on its own Docker agent so the controller needs no pre-installed toolchain.
pipeline {
    agent none

    stages {
        stage('Run Pytest Suite') {
            agent { docker { image 'python:3.11-slim' } }
            steps {
                sh 'pip install --quiet -r requirements.txt'
                sh 'python -m pytest tests/ -v'
            }
        }

        stage('Build and test the TypeScript feature-mappings validator') {
            agent { docker { image 'node:22-slim' } }
            steps {
                dir('tools/validate-feature-mappings') {
                    sh 'npm install'
                    sh 'npm run build'
                    sh 'npm test'
                    sh 'npm run validate'
                }
            }
        }
    }
}
