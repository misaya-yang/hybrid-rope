#!/usr/bin/env python3
"""
Needle-in-a-Haystack 合成数据生成工具
用于测试远距离关键 token 的注意力保留能力
"""

import random
import string
from typing import List, Tuple, Optional
import torch
from transformers import GPT2Tokenizer


class NeedleInHaystackGenerator:
    """
    生成 Needle-in-a-Haystack 测试数据
    
    在随机文本(haystack)中插入特定的关键字符串(needle)，
    用于测试模型在远距离位置保留关键信息的能力。
    """
    
    DEFAULT_NEEDLES = [
        "MAGIC_NEEDLE_12345",
        "SECRET_TOKEN_XYZ789",
        "THE_QUICK_BROWN_FOX_JUMPS",
        "ANSWER_IS_FORTY_TWO",
    ]
    
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        seq_len: int = 4096,
        needle_tokens: Optional[List[str]] = None
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.needles = needle_tokens or self.DEFAULT_NEEDLES
        
        # 预编码所有 needle
        self.needle_encodings = {
            needle: tokenizer.encode(needle, add_special_tokens=False)
            for needle in self.needles
        }
        
    def generate_random_text(self, n_tokens: int) -> str:
        """
        生成随机文本作为 haystack
        使用常见英文单词组合，使文本看起来更自然
        """
        # 常见英文单词列表
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have",
            "I", "it", "for", "not", "on", "with", "he", "as", "you",
            "do", "at", "this", "but", "his", "by", "from", "they",
            "we", "say", "her", "she", "or", "an", "will", "my",
            "one", "all", "would", "there", "their", "what", "so",
            "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just",
            "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than",
            "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how",
            "our", "work", "first", "well", "way", "even", "new",
            "want", "because", "any", "these", "give", "day", "most",
            "us", "is", "was", "are", "were", "been", "has", "had",
            "did", "does", "doing", "done", "being", "having"
        ]
        
        # 生成随机句子
        words = []
        while len(words) < n_tokens * 2:  # 多生成一些，因为tokenizer会压缩
            sentence_len = random.randint(5, 15)
            sentence_words = random.choices(common_words, k=sentence_len)
            sentence = " ".join(sentence_words) + "."
            words.extend(sentence.split())
        
        text = " ".join(words[:n_tokens * 2])
        return text
    
    def create_needle_sequence(
        self,
        needle: str,
        needle_position: int,
        haystack_text: Optional[str] = None
    ) -> Tuple[torch.Tensor, Tuple[int, int], str]:
        """
        创建包含 needle 的序列
        
        Args:
            needle: 要插入的 needle 字符串
            needle_position: needle 插入位置 (token index)
            haystack_text: 自定义 haystack 文本，None则随机生成
            
        Returns:
            (token_ids, (needle_start, needle_end), full_text)
        """
        needle_ids = self.needle_encodings[needle]
        needle_len = len(needle_ids)
        
        # 确保有足够空间
        if needle_position + needle_len > self.seq_len:
            needle_position = self.seq_len - needle_len - 1
        
        # 生成或准备 haystack
        if haystack_text is None:
            # 需要生成足够的随机文本
            haystack_text = self.generate_random_text(self.seq_len)
        
        # Tokenize haystack
        haystack_ids = self.tokenizer.encode(haystack_text, add_special_tokens=False)
        
        # 构建序列: [haystack_prefix] + [needle] + [haystack_suffix]
        prefix_len = min(needle_position, len(haystack_ids))
        suffix_space = self.seq_len - prefix_len - needle_len
        suffix_len = min(suffix_space, len(haystack_ids) - prefix_len)
        
        sequence_ids = (
            haystack_ids[:prefix_len] +
            needle_ids +
            haystack_ids[prefix_len:prefix_len + suffix_len]
        )
        
        # 记录 needle 实际位置
        needle_start = prefix_len
        needle_end = prefix_len + needle_len
        
        # 截断或填充到 seq_len
        if len(sequence_ids) > self.seq_len:
            sequence_ids = sequence_ids[:self.seq_len]
        else:
            # 用 eos_token 填充
            pad_id = self.tokenizer.eos_token_id or 0
            sequence_ids.extend([pad_id] * (self.seq_len - len(sequence_ids)))
        
        # 构建完整文本（用于调试）
        full_text = (
            self.tokenizer.decode(haystack_ids[:prefix_len]) +
            f" [{needle}] " +
            self.tokenizer.decode(haystack_ids[prefix_len:prefix_len + suffix_len])
        )
        
        return torch.tensor(sequence_ids, dtype=torch.long), (needle_start, needle_end), full_text
    
    def create_batch_with_varying_distance(
        self,
        needle: Optional[str] = None,
        distances: Optional[List[int]] = None,
        haystack_text: Optional[str] = None
    ) -> List[Tuple[torch.Tensor, int, str]]:
        """
        创建一批不同 needle 距离的序列
        
        Returns:
            List of (token_ids, distance, needle)
        """
        if needle is None:
            needle = self.needles[0]
        
        if distances is None:
            # 默认测试距离
            distances = [100, 500, 1000, 2000, 3000, 4000]
        
        # 过滤超出范围的距离
        needle_len = len(self.needle_encodings[needle])
        valid_distances = [
            d for d in distances 
            if d + needle_len < self.seq_len
        ]
        
        batch = []
        for distance in valid_distances:
            # needle_position = seq_len - distance - needle_len
            needle_position = self.seq_len - distance - needle_len
            
            token_ids, (start, end), text = self.create_needle_sequence(
                needle, max(0, needle_position), haystack_text
            )
            
            # 重新计算实际距离
            actual_distance = self.seq_len - end
            
            batch.append((token_ids, actual_distance, needle))
        
        return batch
    
    def create_multiple_needles_sequence(
        self,
        needles_with_positions: List[Tuple[str, int]],
        haystack_text: Optional[str] = None
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, str]]]:
        """
        创建包含多个 needle 的序列
        
        Args:
            needles_with_positions: List of (needle, position)
            haystack_text: 自定义 haystack
            
        Returns:
            (token_ids, list of (start, end, needle))
        """
        if haystack_text is None:
            haystack_text = self.generate_random_text(self.seq_len)
        
        haystack_ids = self.tokenizer.encode(haystack_text, add_special_tokens=False)
        
        # 按位置排序 needle
        sorted_needles = sorted(needles_with_positions, key=lambda x: x[1])
        
        sequence_ids = []
        needle_locations = []
        current_pos = 0
        
        for needle, target_pos in sorted_needles:
            needle_ids = self.needle_encodings[needle]
            
            # 确保位置有效
            insert_pos = min(target_pos, self.seq_len - len(needle_ids))
            
            # 添加 haystack 到 insert_pos
            if insert_pos > current_pos:
                sequence_ids.extend(haystack_ids[current_pos:insert_pos])
                current_pos = insert_pos
            
            # 添加 needle
            needle_start = len(sequence_ids)
            sequence_ids.extend(needle_ids)
            needle_end = len(sequence_ids)
            needle_locations.append((needle_start, needle_end, needle))
            
            current_pos = insert_pos
        
        # 添加剩余 haystack
        remaining = self.seq_len - len(sequence_ids)
        if remaining > 0 and current_pos < len(haystack_ids):
            sequence_ids.extend(haystack_ids[current_pos:current_pos + remaining])
        
        # 填充到 seq_len
        if len(sequence_ids) < self.seq_len:
            pad_id = self.tokenizer.eos_token_id or 0
            sequence_ids.extend([pad_id] * (self.seq_len - len(sequence_ids)))
        else:
            sequence_ids = sequence_ids[:self.seq_len]
        
        return torch.tensor(sequence_ids, dtype=torch.long), needle_locations


def test_needle_generator():
    """测试 Needle 生成器"""
    from transformers import GPT2Tokenizer
    
    print("Testing NeedleInHaystackGenerator...")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    generator = NeedleInHaystackGenerator(tokenizer, seq_len=512)
    
    # 测试1: 基本序列生成
    print("\n1. Basic sequence generation:")
    token_ids, (start, end), text = generator.create_needle_sequence(
        "MAGIC_NEEDLE_12345", 100
    )
    print(f"   Sequence length: {len(token_ids)}")
    print(f"   Needle position: {start} to {end}")
    print(f"   Needle text preview: ...{text[max(0, start-20):end+20]}...")
    
    # 验证 needle 位置
    decoded = tokenizer.decode(token_ids[start:end])
    print(f"   Decoded needle: '{decoded}'")
    assert "MAGIC_NEEDLE" in decoded, "Needle not found at expected position!"
    
    # 测试2: 不同距离批次
    print("\n2. Batch with varying distances:")
    batch = generator.create_batch_with_varying_distance(
        distances=[50, 100, 200, 400]
    )
    for token_ids, distance, needle in batch:
        print(f"   Distance {distance}: shape={token_ids.shape}")
    
    # 测试3: 多 needle 序列
    print("\n3. Multiple needles sequence:")
    needles_pos = [
        ("MAGIC_NEEDLE_12345", 100),
        ("SECRET_TOKEN_XYZ789", 300),
    ]
    token_ids, locations = generator.create_multiple_needles_sequence(needles_pos)
    print(f"   Sequence length: {len(token_ids)}")
    for start, end, needle in locations:
        decoded = tokenizer.decode(token_ids[start:end])
        print(f"   Needle at {start}-{end}: '{decoded}'")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_needle_generator()
