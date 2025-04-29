from metaflow import FlowSpec, step


class LinearFlow(FlowSpec):
    @step
    def start(self):
        self.my_var = "hello world"
        self.next(self.a)

    @step
    def a(self):
        print("the data artifact is: %s" % self.my_var)
        import subprocess
        # TODO(mahdi): create a volume for data and copy them
        cmd = [
            "docker", "run",
            "--rm",
            # tests volume
            "-v", "/Users/mahdikhashan/tmp/tu-mlops/tests:/tests",
            # data volume
            "-v", "/Users/mahdikhashan/tmp/tu-mlops/data:/tests/data",
            "-w", "/",
            # "--env", env_var,       # Pass specific filename for this run
            "my-local-test-env:latest",
            "sh", "-c", "python -m pytest -v tests"
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Docker run stdout (contains pytest output):")
            print(result.stdout)
            print("Docker run stderr:")
            print(result.stderr)
            print("---- Validation SUCCEEDED ----")
            self.validation_passed = True
        except subprocess.CalledProcessError as e:
            print(f"Error: Docker run command failed with exit code {e.returncode}")
            print("Docker run stdout:")
            print(e.stdout)
            print("Docker run stderr:")
            print(e.stderr)
            self.validation_passed = False
            raise ValueError(f"Pytest validation inside Docker failed! Check Docker output above.") from e
        except FileNotFoundError:
             print(f"Error: 'docker' command not found. Is Docker installed and in your PATH?")
             raise
        except Exception as e:
             print(f"An unexpected error occurred: {e}")
             raise

        self.next(self.end)

    @step
    def end(self):
        print("the data artifact is still: %s" % self.my_var)


if __name__ == "__main__":
    LinearFlow()
