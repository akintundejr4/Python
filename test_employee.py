import unittest
from employee import Employee
from unittest.mock import patch

class TestEmployee(unittest.TestCase):
    """Test class for working through unittest features"""

    @classmethod
    def setUpClass(cls):
        print('setupClass')

    @classmethod
    def tearDownClass(cls):
        print('teardownClass')

    def setUp(self):
        print('setUp')
        self.emp_1 = Employee('Sasuke', 'Uchia', 50000)
        self.emp_2 = Employee('Mikasa', 'Ackermann', 60000)

    def tearDown(self):
        print('tearDown\n')

    def test_email(self):
        print('test_email')
        self.assertEqual(self.emp_1.email, 'Sasuke.Uchia@email.com')
        self.assertEqual(self.emp_2.email, 'Mikasa.Ackermann@email.com')

        self.emp_1.first = 'John'
        self.emp_2.first = 'Jane'

        self.assertEqual(self.emp_1.email, 'John.Uchia@email.com')
        self.assertEqual(self.emp_2.email, 'Jane.Ackermann@email.com')

    def test_fullname(self):
        print('test_fullname')
        self.assertEqual(self.emp_1.fullname, 'Sasuke Uchia')
        self.assertEqual(self.emp_2.fullname, 'Mikasa Ackermann')

        self.emp_1.first = 'John'
        self.emp_2.first = 'Jane'

        self.assertEqual(self.emp_1.fullname, 'John Uchia')
        self.assertEqual(self.emp_2.fullname, 'Jane Ackermann')

    def test_apply_raise(self):
        print('test_apply_raise')
        self.emp_1.apply_raise()
        self.emp_2.apply_raise()

        self.assertEqual(self.emp_1.pay, 52500)
        self.assertEqual(self.emp_2.pay, 63000)

if __name__ == '__main__':
    # Allows for running tests as python -m unittest test_employee.py
    unittest.main()