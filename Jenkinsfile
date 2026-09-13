// Mirrors .github/workflows/ci.yml (Python test suite + TypeScript validator build/test),
// each stage on its own Docker agent so the controller needs no pre-installed toolchain.
pipeline {
    agent none

    stages {
        stage('Run Pytest Suite') {
            // -u root: the mapped Jenkins UID has no passwd entry in this image, which
            // breaks pip's HOME-relative cache/user-site paths otherwise.
            agent { docker { image 'python:3.11-slim'; args '-u root:root' } }
            steps {
                // pyodbc needs libodbc.so.2 at import time; python:3.11-slim doesn't
                // ship it (ubuntu-latest's GitHub Actions runner has it preinstalled,
                // which is why ci.yml's requirements.txt alone is enough there).
                sh 'apt-get update -qq && apt-get install -y -qq unixodbc'
                sh 'pip install --quiet -r requirements.txt'
                sh 'python -m pytest tests/ -v'
            }
        }

        stage('Build and test the TypeScript feature-mappings validator') {
            agent { docker { image 'node:22-slim'; args '-u root:root' } }
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
