SOM4Proteins
============

## Running tests

1. Download the [zip containing the test files](https://dl.dropboxusercontent.com/u/34915384/testfiles.zip)

2. Unzip the file inside the `SOM4Proteins/test/` directory

3. Create `SOM4Proteins/test/output/` directory

4. Run the following two commands (replace `/absolute/path/to/project/directory` with the path where the project is located on your file system):
```
$ export PROJECT_ABS_DIR=/absolute/path/to/project/directory
$ nosetest3 test/som4proteins
```

## Building documentation

```
$ cd doc
$ make html
```

To view the documentation open `doc/_build/html/index.html`.
