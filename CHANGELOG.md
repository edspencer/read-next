# read-next

## 0.4.0

### Minor Changes

- b5d81d9: [Experimental] Added initial meta-content generation via .prompt(). API will likely change
- b5d81d9: Allows passing in sourceDocuments at read-next creation time or generation time
- b5d81d9: Allow passing summarizationPrompt directly as an argument to index()
- b5d81d9: Added getSourceDocument callback

## 0.3.0

### Minor Changes

- 56dee1d: Allow indexing to be run in parallel

## 0.2.0

### Minor Changes

- 84ee67b: colorful cache miss/hit and expensive operation hints in log output

### Patch Changes

- 84ee67b: Refactor ContentHasher and logger out, add tests
- 84ee67b: Better handling for forward slashes in ids
- 84ee67b: create a subdir inside os.tmpdir if no cacheDir given

## 0.1.1

### Patch Changes

- 5d93f55: JS Doc
- 5d93f55: Round out test coverage

## 0.1.0

### Minor Changes

- 1f1ac25: README and cleanup

### Patch Changes

- 7dbe7d2: Fix duplicates issue when you add the same document ID multiple times
- 42d1af1: Use winston for logging
- a8575c7: Fix duplicates issue when you add the same document ID multiple times

## 0.0.3

### Patch Changes

- 7da5d33: clean up dependencies

## 0.0.2

### Patch Changes

- 62cf38a: Fix test running in CI
