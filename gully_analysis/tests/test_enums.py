import unittest

from gully_analysis.enums import Assets, Displayable


class TestEnums(unittest.TestCase):
    def test_display_name(self):
        class TestEnum(Displayable):
            TEST_ONE = 'testone'
            TESTTWO = 'testtwo'

        self.assertEqual(TestEnum.TEST_ONE.display_name(), 'Test One')
        self.assertEqual(TestEnum.TESTTWO.display_name(), 'Testtwo')

    def test_assets(self):
        for path in Assets:
            self.assertTrue(path.value.exists())
