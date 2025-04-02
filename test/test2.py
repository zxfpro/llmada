import pytest

def sum(a,b):
    return a + b


class TestDemo:
    @pytest.mark.skip(reason='暂时不执行')
    def test_case1(self):
        assert sum(1,4) == 5
    def test_case2(self):
        assert sum(1,4) == 3

class TestDemo2:
    def test_case3(self):
        assert sum(1,4) == 5
    def test_case4(self):
        assert sum(1,4) == 35
if __name__ == '__main__':
    pytest.main()

