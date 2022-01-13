# ABM-project8
Agent Based Modeling project of group 8

students with student number:
- Maurits Bos: 14014777
- Noah van de Bunt: 11226218
- Sasha Boykova: XXXXXXX
- Nina Holzbach: 13827464
- Johanna : 12108006

## workflow 
main points of attention:
- Please use vscode as text editor 😅.
- If you make edits to main files you should create a branch with git and then if you think its working create a merge request. 
- Try to work with `.py` files and not with `.ipynb`. 
- Try to isolate functionality in functions as much as possible (see example below)
- write tests in the docstrings of the functions 


## filestructure
- Write all the python code for the project in `./src/` 
- See ./tests/. For complex code we should write tests. to test functions from `example.py` should be named `test_example.py`
- use folder file for random stuf



## examples

### function isolation
I know its a bit silly examplpe but you get the idea. The most important concequence of function isolation is that your code is better readable by your team mates, which of course contributes to the quality of the project! 💪 
```python
  rnd = np.random.random()
  x = 100
  y = 10
  z = 200
  
  if rnd > 0.5:
     u = x^2
     z = z + u * rnd
   else:
    u = y^2
    z = z + u * rnd  
```

its  better to write:
```python
  rnd = np.random.random()
  x = 100
  y = 10
  z = 200
  
  def calculate(z, u, rnd):
    return (z + x^2) * rnd
  
  if rnd > 0.5: 
    z = calculate(z,x,rnd)
   
   else:
    z = calculate(z,y,rnd)
```

### test using docstring:
Using tests in docstring prevents making simple mistakes which bite your ass later on.
in this example the function should multiply, but i made a simple typo...
because of the test, it'll automatically do a sanity check and filter out the mistaces.

The testing works as follows: the test evaluates this line `>>>func(2,2,2)` and it succeeds if the value equals `8`. 

```python
  def func(x, y, z):
    """ This function should multiply all the given parameters
    
    >>> func(2, 2, 2) 
    8
    """
    
    return x * y + z
```
