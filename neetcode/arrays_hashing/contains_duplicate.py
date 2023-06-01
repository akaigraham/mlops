# given an integer array nums, return true if any value appears at least twice in the 
# array, and return false if every element is distinct

class Solution:
	def containsDuplicate(self, nums: List[int]) -> bool:

		counts_dict = {}
		for num in nums:
			try:
				counts_dict[num]
				return True
			except KeyError:
				counts_dict[num] = 1
		return False