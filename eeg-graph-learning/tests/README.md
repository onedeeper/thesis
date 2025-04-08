# Testing Documentation

I implemented testing for the functions I wrote with Pytest. When I started the project, I wanted to get a move on so I decided to utilize an example file from the sample data. However, in thinking more about this, I realize this might have not been a good idea.

## Thinking about the testing scripts

### Current Limitations
- Using a specific example file from dataset as a test case might not be a good idea
- I want to test a given function or method in general for all possible instances of an object.
- As it stands, I test it on a specific instance which makes the tests somewhat useful, but not complete.

### Proposed Improvements
- To improve on this, I would have to set up a better way of generating a mock example of the data which a function is expected to process:
  - Start with variations of eeg_data CSVs
  - Load these and create MNE objects and continue with the tests

### Additional Notes
- I find that asserts that I use in the code make some of these tests redundant
- I will include these tests for now, and I will add a readme reflecting on this