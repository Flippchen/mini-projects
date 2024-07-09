class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        from collections import Counter

        if not s or not words:
            return []

        word_length = len(words[0])
        num_words = len(words)
        substring_length = word_length * num_words
        words_counter = Counter(words)
        result = []

        for i in range(word_length):
            left = i
            right = i
            current_counter = Counter()
            while right + word_length <= len(s):
                word = s[right:right + word_length]
                right += word_length
                if word in words_counter:
                    current_counter[word] += 1
                    while current_counter[word] > words_counter[word]:
                        current_counter[s[left:left + word_length]] -= 1
                        left += word_length
                    if right - left == substring_length:
                        result.append(left)
                else:
                    current_counter.clear()
                    left = right

        return result
