import calc
import unittest

class TestCalc(unittest.TestCase): 

    def test_add(self): 
        self.assertEqual(calc.add(-1, 1), 0)
        self.assertEqual(calc.add(-1, -1), -2)
        self.assertEqual(calc.add(10, 5), 15)

    def test_subtract(self): 
        self.assertEqual(calc.subtract(-1, 1), -2)
        self.assertEqual(calc.subtract(-1, -1), 0)
        self.assertEqual(calc.subtract(10, 5), 5)       
    
    def test_multiply(self): 
        self.assertEqual(calc.multiply(-1, 1), -1)
        self.assertEqual(calc.multiply(-1, -1), 1)
        self.assertEqual(calc.multiply(10, 5), 50)

    def test_divide(self): 
        self.assertEqual(calc.divide(-1, 1), -1)
        self.assertEqual(calc.divide(-1, -1), 1)
        self.assertEqual(calc.divide(10, 5), 2)
        
        with self.assertRaises(ValueError): 
            calc.divide(10, 0)
            calc.divide(129029920, 0)


if __name__ == "__main__": 
    # Allows for running tests as python -m unittest test_calc.py
    unittest.main()