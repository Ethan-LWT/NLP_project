"""
Content quality assessment for NLP text data.

This module provides functions to analyze text quality, detect problematic content,
and filter out undesirable elements to improve dataset quality.
"""

import re
import logging
import math
import collections
from typing import Dict, List, Tuple, Union, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('content_quality')

# Constants for quality thresholds
QUALITY_THRESHOLDS = {
    'high': 0.8,
    'medium': 0.5,
    'low': 0.3
}

class ContentQualityAnalyzer:
    """
    Analyzer for assessing text quality and identifying problematic content.
    """
    
    def __init__(self, language: str = 'auto'):
        """
        Initialize the content quality analyzer.
        
        Args:
            language: Language code ('en', 'zh', etc.) or 'auto' to detect
        """
        self.language = language
        self.boilerplate_patterns = self._get_boilerplate_patterns()
        self.structural_patterns = self._get_structural_patterns()
        
    def _get_boilerplate_patterns(self) -> Dict[str, List[str]]:
        """Get patterns for detecting boilerplate content by language"""
        # Common patterns across languages
        common = [
            r'copyright',
            r'all\s+rights\s+reserved',
            r'powered\s+by',
            r'terms\s+(of\s+)?(use|service)',
            r'privacy\s+policy',
            r'contact\s+us',
            r'follow\s+us',
            r'share\s+this',
            r'subscribe\s+to',
            r'newsletter',
            r'sign\s+up',
            r'log\s+in',
            r'register',
            r'username',
            r'password'
        ]
        
        # Chinese-specific patterns
        chinese = [
            r'版权所有',
            r'保留所有权利',
            r'备案号',
            r'ICP',
            r'网站地图',
            r'关于我们',
            r'联系我们',
            r'使用条款',
            r'隐私政策',
            r'登录',
            r'注册',
            r'用户名',
            r'密码',
            r'欢迎来到',
            r'关注我们',
            r'点击查看',
            r'上一题',
            r'下一题',
            r'目录',
            r'单项选择题',
            r'多项选择题',
            r'判断题',
            r'填空题',
            r'答案',
            r'解析',
            r'在线考试',
            r'题库',
            r'首页',
            r'顶部',
            r'底部'
        ]
        
        # Japanese-specific patterns
        japanese = [
            r'著作権',
            r'無断転載',
            r'禁止',
            r'利用規約',
            r'プライバシーポリシー',
            r'問い合わせ',
            r'ログイン',
            r'新規登録',
            r'ユーザー名',
            r'パスワード',
            r'選択問題',
            r'正しい答え',
            r'解答',
        ]
        
        # Russian-specific patterns
        russian = [
            r'авторские права',
            r'все права защищены',
            r'политика конфиденциальности',
            r'пользовательское соглашение',
            r'вход',
            r'регистрация',
            r'пароль',
            r'контакты',
            r'о сайте',
            r'поиск'
        ]
        
        # Spanish-specific patterns
        spanish = [
            r'derechos reservados',
            r'política de privacidad',
            r'términos de servicio',
            r'iniciar sesión',
            r'registrarse',
            r'contraseña',
            r'contáctenos',
            r'acerca de',
            r'selección múltiple',
            r'verdadero o falso',
            r'respuesta correcta'
        ]
        
        # Arabic-specific patterns
        arabic = [
            r'جميع الحقوق محفوظة',
            r'حقوق النشر',
            r'سياسة الخصوصية',
            r'شروط الاستخدام',
            r'تسجيل الدخول',
            r'اشتراك',
            r'كلمة المرور',
            r'اتصل بنا',
            r'من نحن'
        ]
        
        # Korean-specific patterns
        korean = [
            r'저작권',
            r'모든 권리 보유',
            r'개인정보 보호정책',
            r'이용약관',
            r'로그인',
            r'회원가입',
            r'비밀번호',
            r'문의하기',
            r'회사소개'
        ]
        
        return {
            'common': common,
            'zh': chinese,
            'ja': japanese,
            'ru': russian,
            'es': spanish,
            'ar': arabic,
            'ko': korean
        }
    
    def _get_structural_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Get patterns for identifying document structure"""
        patterns = {
            'common': {
                'test_question': [
                    r'^[0-9]+\.\s',  # Numbered questions
                    r'^Q[0-9]+:',    # Q1:, Q2:, etc.
                    r'^Question\s+[0-9]+',  # Question 1, Question 2
                ],
                'multiple_choice': [
                    r'^[A-D][\s.)\]]+[a-zA-Z]',  # A. option, B) option
                    r'^[0-9]+[).]\s+[A-D][).]\s+'  # 1) A) option
                ]
            },
            'zh': {
                'test_question': [
                    r'^[0-9]+[\s.、：:]+',  # Chinese numbered questions
                    r'^第[一二三四五六七八九十百千万]+题',  # Chinese character numbers
                    r'^题目[\s:：]'  # "Question:" in Chinese
                ],
                'multiple_choice': [
                    r'^[ABCD][\s.、：:]+',  # Chinese multiple choice
                    r'（?[ABCD]）',  # Options in parentheses
                ]
            },
            'ja': {
                'test_question': [
                    r'^[0-9]+[\s.、：:]+',  # Japanese numbered questions
                    r'^問題[0-9]+',  # "Question X" in Japanese
                    r'^第[一二三四五六七八九十]+問'  # Japanese character numbers
                ],
                'multiple_choice': [
                    r'^[ABCD][\s.、：:]+',  # Japanese multiple choice
                    r'(?:[あいうえお])',  # Options with Japanese characters
                    r'(?:[アイウエオ])'  # Options with katakana
                ]
            },
            'ru': {
                'test_question': [
                    r'^[0-9]+[\s.)\]]+',  # Russian numbered questions
                    r'^Вопрос\s+[0-9]+',  # "Question X" in Russian
                    r'^Задание\s+[0-9]+'  # "Task X" in Russian
                ],
                'multiple_choice': [
                    r'^[АБВГ][\s.)\]]+',  # Russian multiple choice
                    r'^[А-Г][\s.)\]]+'  # Russian multiple choice (range)
                ]
            },
            'es': {
                'test_question': [
                    r'^[0-9]+[\s.)\]]+',  # Spanish numbered questions
                    r'^Pregunta\s+[0-9]+',  # "Question X" in Spanish
                    r'^Cuestión\s+[0-9]+'  # "Task X" in Spanish
                ],
                'multiple_choice': [
                    r'^[ABCD][\s.)\]]+',  # Spanish multiple choice
                ]
            },
            'ar': {
                'test_question': [
                    r'^[0-9]+[\s.)\]]+',  # Arabic numbered questions
                    r'^سؤال\s+[0-9]+',  # "Question X" in Arabic
                ],
                'multiple_choice': [
                    r'^[أبجد][\s.)\]]+',  # Arabic multiple choice
                    r'^[ابجد][\s.)\]]+'  # Arabic multiple choice alternatives
                ]
            },
            'ko': {
                'test_question': [
                    r'^[0-9]+[\s.)\]]+',  # Korean numbered questions
                    r'^문제\s+[0-9]+',  # "Question X" in Korean
                ],
                'multiple_choice': [
                    r'^[ABCD][\s.)\]]+',  # Korean multiple choice (often uses Latin alphabet)
                    r'(?:[가나다라])'  # Options with Korean characters
                ]
            }
        }
        return patterns
    
    def analyze_text(self, text: str, deep_analysis: bool = False) -> Dict[str, Any]:
        """
        Analyze text and return quality metrics.
        
        Args:
            text: Text to analyze
            deep_analysis: Whether to perform more comprehensive analysis
            
        Returns:
            Dictionary with quality metrics
        """
        # Basic metrics
        if not text or len(text) < 10:
            return {'quality_score': 0.0, 'reason': 'Text too short or empty'}
        
        metrics = {
            'length': len(text),
            'line_count': text.count('\n') + 1,
            'word_count': len(re.findall(r'\b\w+\b', text)),
        }
        
        # Detect language if set to auto
        if self.language == 'auto':
            import importlib
            if importlib.util.find_spec('text_cleaning') is not None:
                from text_cleaning import detect_language
                lang_info = detect_language(text[:5000])
                detected_language = lang_info.get('language', 'unknown')
            else:
                # Simple language detection as fallback
                detected_language = self._detect_language_simple(text)
            metrics['language'] = detected_language
        else:
            metrics['language'] = self.language
        
        # Quality indicators
        metrics.update(self._calculate_quality_indicators(text, metrics['language']))
        
        # Calculate overall quality score
        metrics['quality_score'] = self._calculate_quality_score(metrics)
        
        # Determine quality category
        if metrics['quality_score'] >= QUALITY_THRESHOLDS['high']:
            metrics['quality_category'] = 'high'
        elif metrics['quality_score'] >= QUALITY_THRESHOLDS['medium']:
            metrics['quality_category'] = 'medium'
        elif metrics['quality_score'] >= QUALITY_THRESHOLDS['low']:
            metrics['quality_category'] = 'low'
        else:
            metrics['quality_category'] = 'very_low'
        
        # Deep analysis if requested
        if deep_analysis:
            metrics.update(self._perform_deep_analysis(text, metrics['language']))
        
        return metrics
    
    def _detect_language_simple(self, text: str) -> str:
        """Simple language detection based on character frequencies"""
        # Take a sample for efficiency
        sample = text[:2000]
        
        # Check for Chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', sample))
        if chinese_chars > 100:
            return 'zh'
            
        # Check for Japanese characters (Hiragana and Katakana)
        japanese_chars = len(re.findall(r'[\u3040-\u30ff]', sample))
        if japanese_chars > 50:
            return 'ja'
            
        # Check for Korean characters (Hangul)
        korean_chars = len(re.findall(r'[\uac00-\ud7a3]', sample))
        if korean_chars > 50:
            return 'ko'
            
        # Check for Russian characters
        russian_chars = len(re.findall(r'[\u0400-\u04FF]', sample))
        if russian_chars > 50:
            return 'ru'
            
        # Default to English for Latin-script text
        return 'en'
    
    def _calculate_quality_indicators(self, text: str, language: str) -> Dict[str, float]:
        """Calculate various quality indicators for the text"""
        indicators = {}
        
        # 1. Text diversity (vocabulary richness)
        words = re.findall(r'\b\w+\b', text.lower())
        if words:
            unique_words = set(words)
            indicators['lexical_diversity'] = len(unique_words) / len(words)
            
            # Calculate word frequency distribution
            word_freq = collections.Counter(words)
            most_common_count = sum(count for word, count in word_freq.most_common(5))
            indicators['top5_concentration'] = most_common_count / len(words)
        else:
            indicators['lexical_diversity'] = 0
            indicators['top5_concentration'] = 1
        
        # 2. Structural metrics
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            # Average line length
            avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
            indicators['avg_line_length'] = avg_line_length
            
            # Short line ratio (often indicative of menus, lists, etc.)
            short_lines = sum(1 for line in non_empty_lines if len(line.strip()) < 40)
            indicators['short_line_ratio'] = short_lines / len(non_empty_lines)
            
            # Very long line ratio (often indicative of machine-generated text)
            long_lines = sum(1 for line in non_empty_lines if len(line.strip()) > 300)
            indicators['long_line_ratio'] = long_lines / len(non_empty_lines)
        else:
            indicators['avg_line_length'] = 0
            indicators['short_line_ratio'] = 1
            indicators['long_line_ratio'] = 0
        
        # 3. Boilerplate detection
        # Get patterns appropriate for the language
        pattern_lists = ['common']
        if language in self.boilerplate_patterns:
            pattern_lists.append(language)
        
        boilerplate_matches = 0
        for pattern_list in pattern_lists:
            for pattern in self.boilerplate_patterns.get(pattern_list, []):
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                boilerplate_matches += matches
        
        indicators['boilerplate_density'] = min(1.0, boilerplate_matches / max(1, len(text) / 1000))
        
        # 4. HTML remnant detection
        html_matches = len(re.findall(r'</?[a-z]+>|&[a-z]+;', text, re.IGNORECASE))
        indicators['html_density'] = min(1.0, html_matches / max(1, len(text) / 100))
        
        # 5. Test/exam content detection
        test_question_patterns = []
        multiple_choice_patterns = []
        
        # Add patterns for the appropriate language
        if language in self.structural_patterns:
            test_question_patterns.extend(self.structural_patterns[language].get('test_question', []))
            multiple_choice_patterns.extend(self.structural_patterns[language].get('multiple_choice', []))
        
        # Add common patterns
        test_question_patterns.extend(self.structural_patterns['common'].get('test_question', []))
        multiple_choice_patterns.extend(self.structural_patterns['common'].get('multiple_choice', []))
        
        question_count = 0
        choice_count = 0
        
        # Count matches for each pattern
        for pattern in test_question_patterns:
            question_count += len(re.findall(pattern, text, re.MULTILINE))
        
        for pattern in multiple_choice_patterns:
            choice_count += len(re.findall(pattern, text, re.MULTILINE))
        
        indicators['question_density'] = min(1.0, question_count / max(1, len(non_empty_lines)))
        indicators['choice_density'] = min(1.0, choice_count / max(1, len(non_empty_lines)))
        
        # Exam content indicator (higher means more likely to be exam content)
        if question_count > 1 and choice_count > 1:
            indicators['exam_content_likelihood'] = min(1.0, (question_count * choice_count) / (len(non_empty_lines) ** 2))
        else:
            indicators['exam_content_likelihood'] = 0.0
        
        return indicators
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score based on metrics"""
        # Weights for different factors
        weights = {
            'lexical_diversity': 0.25,  # Higher diversity is better
            'top5_concentration': -0.10,  # Lower concentration is better
            'avg_line_length': 0.05,  # Medium line length is ideal
            'short_line_ratio': -0.10,  # Fewer short lines is better
            'long_line_ratio': -0.05,  # Fewer very long lines is better
            'boilerplate_density': -0.20,  # Less boilerplate is better
            'html_density': -0.10,  # Less HTML remnants is better
            'question_density': -0.05,  # Fewer question patterns is better
            'choice_density': -0.05,  # Fewer multiple choice patterns is better
            'exam_content_likelihood': -0.10,  # Lower exam likelihood is better
        }
        
        # Base score
        score = 0.5
        
        # Apply weighted factors
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                
                # Special handling for avg_line_length
                if metric == 'avg_line_length':
                    # Optimal line length around 80-120 characters
                    if 40 <= value <= 200:
                        # Normalize to 0-1 with peak at 120
                        normalized_value = 1.0 - abs(value - 120) / 120
                        score += weight * normalized_value
                    else:
                        score += weight * 0.2  # Penalize extreme lengths but not too heavily
                else:
                    # For other metrics, apply weight directly
                    score += weight * value
        
        # Ensure score is in 0-1 range
        return max(0.0, min(1.0, score))
    
    def _perform_deep_analysis(self, text: str, language: str) -> Dict[str, Any]:
        """Perform deeper content analysis for better quality assessment"""
        deep_metrics = {}
        
        # Split into paragraphs for analysis
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return {'deep_analysis': 'failed', 'reason': 'No paragraphs found'}
        
        # Calculate paragraph length statistics
        para_lengths = [len(p) for p in paragraphs]
        deep_metrics['avg_paragraph_length'] = sum(para_lengths) / len(paragraphs)
        deep_metrics['paragraph_count'] = len(paragraphs)
        
        # Calculate paragraph similarity (repetition detection)
        if len(paragraphs) > 1:
            # Use a simplified similarity check
            from collections import Counter
            
            # Create paragraph fingerprints (first 50 chars)
            fingerprints = [p[:50].lower() for p in paragraphs if len(p) >= 50]
            if fingerprints:
                fingerprint_counts = Counter(fingerprints)
                # Calculate repetition ratio
                deep_metrics['paragraph_repetition'] = (
                    sum(count - 1 for count in fingerprint_counts.values()) / 
                    max(1, len(fingerprints) - 1)
                )
            else:
                deep_metrics['paragraph_repetition'] = 0.0
        else:
            deep_metrics['paragraph_repetition'] = 0.0
        
        # Check for content quality issues
        deep_metrics['quality_issues'] = self._check_quality_issues(text, language)
        
        # Determine if content should be filtered
        should_filter = (
            deep_metrics['quality_issues'].get('exam_content', False) or
            deep_metrics['quality_issues'].get('excessive_boilerplate', False) or
            deep_metrics['quality_issues'].get('excessive_formatting', False) or
            deep_metrics['quality_issues'].get('machine_generated', False) or
            deep_metrics['quality_issues'].get('highly_repetitive', False)
        )
        deep_metrics['should_filter'] = should_filter
        
        return deep_metrics
    
    def _check_quality_issues(self, text: str, language: str) -> Dict[str, bool]:
        """Check for specific quality issues in the text"""
        issues = {}
        
        # 1. Exam content check - use language-specific patterns
        exam_indicators = {
            'en': [
                r'multiple choice', r'true or false', r'fill in the blank',
                r'select the correct answer', r'choose the best answer'
            ],
            'zh': [
                r'单项选择题', r'多项选择题', r'判断题', r'填空题'
            ],
            'ja': [
                r'選択問題', r'記述問題', r'正しい答え', r'解答'
            ],
            'ru': [
                r'выберите правильный ответ', r'выберите вариант', r'верно или неверно'
            ],
            'es': [
                r'selección múltiple', r'verdadero o falso', r'llene el espacio en blanco',
                r'elija la respuesta correcta'
            ],
            'ar': [
                r'اختيار من متعدد', r'صح أم خطأ', r'املأ الفراغ'
            ],
            'ko': [
                r'객관식', r'주관식', r'맞다 틀리다', r'옳은 답'
            ]
        }
        
        # Apply language-specific patterns plus common ones
        lang_specific_patterns = exam_indicators.get(language, [])
        common_patterns = exam_indicators.get('en', [])  # Use English as fallback
        
        # Check all applicable patterns
        all_patterns = lang_specific_patterns + common_patterns
        exam_count = sum(1 for pattern in all_patterns if re.search(pattern, text, re.IGNORECASE))
        
        # Check multiple choice pattern that works across languages
        has_multiple_choice = re.search(r'[A-D]\..*[A-D]\..*[A-D]\.', text, re.DOTALL) is not None
        
        issues['exam_content'] = exam_count >= 2 or has_multiple_choice
        
        # 2. Excessive boilerplate check - use language-specific patterns
        boilerplate_indicators = {
            'common': [
                r'copyright', r'all rights reserved', r'terms of service', 
                r'privacy policy', r'login', r'register'
            ],
            'zh': [
                r'版权所有', r'保留所有权利', r'备案号', r'ICP', 
                r'登录', r'注册', r'关于我们', r'联系我们'
            ],
            'ja': [
                r'著作権', r'無断転載', r'禁止', r'利用規約', 
                r'プライバシー', r'ログイン', r'新規登録'
            ],
            'ru': [
                r'авторские права', r'все права защищены', 
                r'политика конфиденциальности', r'вход', r'регистрация'
            ],
            'es': [
                r'derechos reservados', r'política de privacidad', 
                r'térминос de servicio', r'iniciar sesión', r'registrarse'
            ],
            'ar': [
                r'حقوق النشر', r'جميع الحقوق محفوظة', 
                r'سياسة الخصوصية', r'تسجيل الدخول', r'اشتراك'
            ],
            'ko': [
                r'저작권', r'모든 권리 보유', r'개인정보 보호정책', 
                r'이용약관', r'로그인', r'회원가입'
            ]
        }
        
        # Get language-specific patterns plus common ones
        lang_specific_boilerplate = boilerplate_indicators.get(language, [])
        common_boilerplate = boilerplate_indicators.get('common', [])
        
        # Check all applicable patterns
        all_boilerplate = lang_specific_boilerplate + common_boilerplate
        boilerplate_count = sum(1 for pattern in all_boilerplate if re.search(pattern, text, re.IGNORECASE))
        
        issues['excessive_boilerplate'] = boilerplate_count >= 3
        
        # 3. Check for excessive HTML/formatting
        formatting_patterns = [r'<[^>]+>', r'&[a-z]+;', r'\[/?[a-z]+\]']
        formatting_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formatting_patterns)
        issues['excessive_formatting'] = formatting_count > len(text) * 0.02  # If >2% is formatting
        
        # 4. Check for machine-generated content
        sentences = re.split(r'[.!?。！？]+', text)
        if len(sentences) > 5:
            # Calculate variance in sentence length
            avg_length = sum(len(s) for s in sentences) / len(sentences)
            variance = sum((len(s) - avg_length) ** 2 for s in sentences) / len(sentences)
            std_dev = math.sqrt(variance)
            
            # Very low standard deviation might indicate machine generation
            issues['machine_generated'] = std_dev < avg_length * 0.3 and avg_length > 20
        else:
            issues['machine_generated'] = False
        
        # 5. Check for highly repetitive content
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) > 50:  # Only check substantive content
            unique_ratio = len(set(words)) / len(words)
            issues['highly_repetitive'] = unique_ratio < 0.3  # Less than 30% unique words
        else:
            issues['highly_repetitive'] = False
        
        return issues

    def should_filter_content(self, text: str) -> Tuple[bool, str]:
        """
        Determine if content should be filtered out based on quality assessment.
        
        Args:
            text: Text content to evaluate
            
        Returns:
            Tuple of (should_filter, reason)
        """
        metrics = self.analyze_text(text)
        
        # Quick checks for obvious cases
        if len(text) < 50:
            return True, "Text too short"
        
        # Calculate various indicators
        has_exam_content = metrics.get('exam_content_likelihood', 0) > 0.5
        has_excessive_boilerplate = metrics.get('boilerplate_density', 0) > 0.3
        has_poor_lexical_diversity = metrics.get('lexical_diversity', 1) < 0.3
        has_html_artifacts = metrics.get('html_density', 0) > 0.2
        quality_score = metrics.get('quality_score', 0)
        
        # Decision logic
        if has_exam_content:
            return True, "Contains exam/test content"
        
        if has_excessive_boilerplate:
            return True, "Excessive boilerplate content"
        
        if has_poor_lexical_diversity and len(text) > 200:
            return True, "Low lexical diversity"
            
        if has_html_artifacts:
            return True, "Contains HTML artifacts"
            
        if quality_score < 0.3:
            return True, f"Low overall quality score: {quality_score:.2f}"
            
        return False, "Content passed quality checks"

def filter_low_quality_content(texts: List[str]) -> List[str]:
    """
    Filter a list of texts to remove low-quality content.
    
    Args:
        texts: List of text strings
        
    Returns:
        List of high-quality text strings
    """
    analyzer = ContentQualityAnalyzer()
    filtered_texts = []
    
    for text in texts:
        should_filter, reason = analyzer.should_filter_content(text)
        if not should_filter:
            filtered_texts.append(text)
    
    return filtered_texts

def analyze_and_clean_document(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Analyze a document's quality and clean it based on analysis results.
    
    Args:
        text: Text to analyze and clean
        
    Returns:
        Tuple of (cleaned_text, quality_metrics)
    """
    analyzer = ContentQualityAnalyzer()
    metrics = analyzer.analyze_text(text, deep_analysis=True)
    
    # Get language to apply appropriate cleaning
    language = metrics.get('language', 'unknown')
    
    # Check if document needs cleaning
    if metrics.get('quality_score', 0) >= QUALITY_THRESHOLDS['high']:
        # Already high quality, perform minimal cleaning
        try:
            from text_cleaning import clean_text
            cleaned_text = clean_text(text, deep_clean=False)
            metrics['cleaning_applied'] = 'minimal'
        except ImportError:
            # If text_cleaning module not available, return original
            cleaned_text = text
            metrics['cleaning_applied'] = 'none'
    else:
        # Lower quality, apply deep cleaning
        try:
            from text_cleaning import clean_text
            cleaned_text = clean_text(text, deep_clean=True)
            
            # For certain languages, apply additional cleaning
            if language in ['zh', 'ja', 'ko']:
                from text_cleaning import clean_cjk_text
                cleaned_text = clean_cjk_text(cleaned_text, language)
                
            metrics['cleaning_applied'] = 'deep'
        except ImportError:
            # If text_cleaning module not available, perform basic cleaning
            cleaned_text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
            cleaned_text = re.sub(r'&[a-z]+;', ' ', cleaned_text)  # Remove HTML entities
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Normalize whitespace
            metrics['cleaning_applied'] = 'basic'
    
    # Analyze quality improvement if cleaning was applied
    if metrics['cleaning_applied'] != 'none':
        after_metrics = analyzer.analyze_text(cleaned_text)
        metrics['quality_before'] = metrics.get('quality_score', 0)
        metrics['quality_after'] = after_metrics.get('quality_score', 0)
        metrics['quality_improvement'] = metrics['quality_after'] - metrics['quality_before']
    
    return cleaned_text, metrics

def batch_analyze_quality(file_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze quality for a batch of files.
    
    Args:
        file_paths: List of file paths to analyze
        
    Returns:
        Dictionary mapping file paths to quality metrics
    """
    results = {}
    analyzer = ContentQualityAnalyzer()
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            results[file_path] = analyzer.analyze_text(text)
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            results[file_path] = {'error': str(e), 'quality_score': 0.0}
    
    return results

def generate_quality_report(results: Dict[str, Dict[str, Any]], output_file: str = None):
    """
    Generate a quality report from analysis results.
    
    Args:
        results: Dictionary mapping file paths to quality metrics
        output_file: File path to write the report (optional)
    """
    report_lines = ["# Content Quality Analysis Report", ""]
    
    # Calculate overall statistics
    quality_scores = [metrics['quality_score'] for metrics in results.values() 
                     if 'quality_score' in metrics]
    
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        report_lines.append(f"## Summary")
        report_lines.append(f"- Files analyzed: {len(results)}")
        report_lines.append(f"- Average quality score: {avg_quality:.2f}")
        report_lines.append(f"- High quality files: {sum(1 for s in quality_scores if s >= QUALITY_THRESHOLDS['high'])}")
        report_lines.append(f"- Medium quality files: {sum(1 for s in quality_scores if QUALITY_THRESHOLDS['medium'] <= s < QUALITY_THRESHOLDS['high'])}")
        report_lines.append(f"- Low quality files: {sum(1 for s in quality_scores if QUALITY_THRESHOLDS['low'] <= s < QUALITY_THRESHOLDS['medium'])}")
        report_lines.append(f"- Very low quality files: {sum(1 for s in quality_scores if s < QUALITY_THRESHOLDS['low'])}")
        report_lines.append("")
    
    # Add file-by-file analysis
    report_lines.append("## File-by-file Analysis")
    
    for file_path, metrics in results.items():
        report_lines.append(f"### {os.path.basename(file_path)}")
        
        if 'error' in metrics:
            report_lines.append(f"- Error: {metrics['error']}")
            continue
            
        quality_score = metrics.get('quality_score', 0)
        quality_category = metrics.get('quality_category', 'unknown')
        
        report_lines.append(f"- Quality Score: {quality_score:.2f} ({quality_category})")
        report_lines.append(f"- Language: {metrics.get('language', 'unknown')}")
        report_lines.append(f"- Length: {metrics.get('length', 0)} characters")
        report_lines.append(f"- Lexical Diversity: {metrics.get('lexical_diversity', 0):.2f}")
        
        # Add quality issues if deep analysis was performed
        if 'quality_issues' in metrics:
            report_lines.append("- Quality Issues:")
            for issue, present in metrics['quality_issues'].items():
                if present:
                    report_lines.append(f"  - {issue.replace('_', ' ').title()}")
        
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Quality report written to {output_file}")
        except Exception as e:
            logger.error(f"Error writing report to {output_file}: {e}")
    
    return report_text

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Analyze text content quality')
    parser.add_argument('file', help='Path to text file to analyze or directory of files')
    parser.add_argument('--deep', action='store_true', help='Perform deep analysis')
    parser.add_argument('--language', default='auto', help='Specify language (default: auto-detect)')
    parser.add_argument('--clean', action='store_true', help='Clean the text based on analysis')
    parser.add_argument('--output', '-o', help='Output file for cleaned text or analysis report')
    parser.add_argument('--batch', action='store_true', help='Process all files in directory')
    parser.add_argument('--pattern', default='*.txt', help='File pattern for batch mode')
    parser.add_argument('--report', action='store_true', help='Generate quality analysis report')
    
    args = parser.parse_args()
    
    try:
        # Process a directory of files
        if os.path.isdir(args.file) or args.batch:
            if not os.path.isdir(args.file):
                logger.error(f"Not a directory: {args.file}")
                sys.exit(1)
            
            import glob
            file_pattern = os.path.join(args.file, args.pattern)
            files = glob.glob(file_pattern)
            
            if not files:
                logger.error(f"No files matching pattern {args.pattern} found in {args.file}")
                sys.exit(1)
            
            logger.info(f"Analyzing {len(files)} files...")
            results = batch_analyze_quality(files)
            
            if args.report:
                report_file = args.output or os.path.join(args.file, "quality_report.md")
                generate_quality_report(results, report_file)
            
        # Process a single file
        else:
            with open(args.file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            analyzer = ContentQualityAnalyzer(language=args.language)
            
            if args.clean:
                cleaned_text, metrics = analyze_and_clean_document(content)
                
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)
                    print(f"Cleaned text written to {args.output}")
                
                print("\nCleaning Summary:")
                print(f"Quality before: {metrics.get('quality_before', 0):.2f}")
                print(f"Quality after: {metrics.get('quality_after', 0):.2f}")
                print(f"Improvement: {metrics.get('quality_improvement', 0):.2f}")
                print(f"Cleaning level: {metrics.get('cleaning_applied', 'none')}")
                
            else:
                results = analyzer.analyze_text(content, deep_analysis=args.deep)
                
                print("\nContent Quality Analysis Results:")
                print(f"Quality Score: {results['quality_score']:.2f} ({results['quality_category']} quality)")
                print(f"Language: {results['language']}")
                print(f"Length: {results['length']} characters, {results['word_count']} words")
                print(f"Lexical Diversity: {results['lexical_diversity']:.2f}")
                print(f"Boilerplate Density: {results['boilerplate_density']:.2f}")
                print(f"HTML Artifact Density: {results['html_density']:.2f}")
                
                if args.deep and 'quality_issues' in results:
                    print("\nQuality Issues:")
                    for issue, present in results['quality_issues'].items():
                        status = "YES" if present else "No"
                        print(f"  {issue.replace('_', ' ').title()}: {status}")
                    
                    filter_decision = results.get('should_filter', False)
                    print(f"\nFilter Recommendation: {'Filter out' if filter_decision else 'Keep'}")
        
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()
