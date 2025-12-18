#!/usr/bin/env bash

#  Copyright (c) 2022 cloudkite.io
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

VERSION="2022.09.27a"

##### terraform.sh #####
# This is an opinionated terraform state manager.
########################
TERRAFORM_BINARY=${TERRAFORM_BINARY:-"$(which terraform)"}

# Maps Environment names to Terraform state buckets
declare -A ENVIRONMENTS
# find path to .env
SCRIPT_PATH="${BASH_SOURCE[0]}"
# Handle symbolic links
if [[ -L "$SCRIPT_PATH" ]]; then
    SCRIPT_PATH=$(readlink "$SCRIPT_PATH")
fi
BASE_PATH=$(dirname "$SCRIPT_PATH")
if [ -f $BASE_PATH/.env ]
then
  set -o allexport
  echo "Loading environment from $BASE_PATH/.env"
  source $BASE_PATH/.env
  set +o allexport
fi

COMMANDS=("apply" "destroy" "import" "init" "plan" "refresh" "state")
TEMPFILES=()
ENVIRONMENT=""

finish() {
  echo ""
  echo "Begin cleanup..."
  for f in "${TEMPFILES[@]}"; do
    echo "Deleting symlink ${f}..."
    rm "${f}"
  done
  for f in "${DISABLEDFILES[@]}"; do
    echo "Changing ${f} back to original name..."
    mv ${f} `basename -s ".disabled" ${f}`
  done
  echo "Done."
  echo ""
}

elementIn() {
  local needle="${1}"
  shift
  local haystack=("${@}")

  for e in "${haystack[@]}"; do
    if [[ "${needle}" == "${e}" ]]; then
      return 0
    fi
  done
  return 1
}

showHelp() {
    local commands=$(echo "${!COMMANDS[@]}>" | tr " " "|")
    local environments=$(echo "${!ENVIRONMENTS[@]}>" | tr " " "|")
    echo "Usage: "
    echo "${0} <${environments}> <${commands}>"
    echo ""
    kill -1 $$
}

symlinkEnvFiles() {
  shopt -s nullglob
  TEMPFILES=()
  for f in *."${ENVIRONMENT}"; do
    new_name="${f}.`date +%s`.tf"
    echo "Temporarily renaming ${f} to ${new_name}"
    ln -s "${f}" "${new_name}"
    TEMPFILES+=(${new_name})
  done
  shopt -u nullglob
}

disableFiles() {
  # This looks for terraform files with a comment that matches:
  # # DISABLED_ENVIRONMENTS: <env1>, <env2>
  # If current env is in the list, we rename the .tf file to append .disabled
  shopt -s nullglob
  DISABLEDFILES=()
  for f in *.tf; do
    if grep --quiet "^# DISABLED_ENVIRONMENTS:.*\s\(${ENVIRONMENT}\).*" ${f}; then
      new_name="${f}.disabled"
      echo "Temporarily renaming disabled ${f} to ${new_name}"
      mv "${f}" "${new_name}"
      DISABLEDFILES+=(${new_name})
    fi
  done
  shopt -u nullglob
}

terraformInit() {
    local env="${1}"
    # Check if .terraform directory exists and has terraform.tfstate
    if [[ -f .terraform/terraform.tfstate ]]; then
        current_prefix=$(cat .terraform/terraform.tfstate | jq -r '.["backend"]["config"]["prefix"]' 2>/dev/null || echo "")
        if [[ ${ENVIRONMENTS[$env]} == "${current_prefix}" ]]; then
            ${TERRAFORM_BINARY} init
            echo "Terraform initialized: env=${env} | prefix=${current_prefix}"
            return
        fi
    fi
    echo "Running terraform init..."
    
    # Check if we're using local backend
    if grep -q 'backend "local"' *.tf; then
        echo "Using local backend, initializing without prefix..."
        ${TERRAFORM_BINARY} init -reconfigure
    else
        echo "Using remote backend, initializing with prefix..."
        ${TERRAFORM_BINARY} init -reconfigure \
            -backend-config="prefix=${ENVIRONMENTS[$env]}"
    fi
    
    if [[ $? -ne 0 ]]; then
      finish
      kill -1 $$
    fi
}

main() {
    ENVIRONMENT="${1}"
    if ! $(elementIn "${ENVIRONMENT}" "${!ENVIRONMENTS[@]}"); then
        echo "Invalid environment specified."
        echo ""
        showHelp
    fi

    symlinkEnvFiles
    disableFiles

    terraformInit "${ENVIRONMENT}"

    shift
    command="${1}"
    if ! $(elementIn "${command}" "${COMMANDS[@]}"); then
        echo "Invalid command specified."
        echo ""
        showHelp
    fi

    shift
    if [[ "${command}" != "init" ]]; then
        echo ""
        echo "Current ${TERRAFORM_BINARY} environment: Environment=${ENVIRONMENT}"
        echo ""
        if [[ "${command}" != "state" ]]; then
          tf_command="${TERRAFORM_BINARY} ${command}"
          for tfvarfile in ./environments/${ENVIRONMENT}/*.tfvars; do
            # support arbitrary tfvarfiles
            tf_command="${tf_command} -var-file=$tfvarfile"
          done
        else tf_command="${TERRAFORM_BINARY} ${command}"
        fi
        eval ${tf_command} '$@'
    fi
}

main $@

trap finish EXIT INT QUIT TERM SIGHUP