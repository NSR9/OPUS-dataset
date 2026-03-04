# OPUS Golden Dataset — Packed Samples Preview

## Overview

| Metric | Value |
|--------|-------|
| Total packed samples | **512** |
| Tokens per sample | **4096** (exact) |
| Total tokens | **2,097,152** |
| Tokenizer | TSAI 131K (vocab 131,072) |
| Domains covered | **50** |
| Raw samples used | 4,291 across 44 HuggingFace datasets |
| Avg padding per sample | 607 tokens |
| Format | `<\|user\|> ... <\|assistant\|> ...` (SFT loss masking ready) |
| Sample separator | `<\|end_of_text\|>` |
| Padding token | `<\|pad\|>` |

## Domain Distribution

| Domain | Packed Samples |
|--------|---------------|
| indic_instruction_native | 81 |
| code_generation | 57 |
| high_reasoning | 41 |
| indic_crosslingual_qa_bengali | 41 |
| instruction_following | 34 |
| math_competition | 28 |
| math_hard | 21 |
| human_preferred | 20 |
| general_knowledge | 17 |
| math_competition_hard | 17 |
| function_calling | 14 |
| general_preference | 13 |
| long_context_multihop | 12 |
| indic_summarization_bn | 11 |
| long_context_qa | 10 |
| general_qa | 8 |
| indic_translation_tamil_en | 7 |
| benchmark_qa | 6 |
| general_conversation | 6 |
| indic_instruction | 6 |
| indic_translation_telugu_en | 6 |
| indic_summarization_hi | 5 |
| indic_translation_bengali_en | 5 |
| indic_translation_hindi_en | 4 |
| long_context_narrative | 4 |
| math_olympiad | 4 |
| indic_fairness | 3 |
| linguistic_diagnostics | 3 |
| long_context_summarization | 3 |
| indic_cultural_telugu | 2 |
| indic_knowledge_tamil | 2 |
| indic_translation_kannada_en | 2 |
| indic_translation_malayalam_en | 2 |
| indic_cultural_bengali | 1 |
| indic_cultural_kannada | 1 |
| indic_cultural_malayalam | 1 |
| indic_cultural_odia | 1 |
| indic_cultural_punjabi | 1 |
| indic_knowledge_bengali | 1 |
| indic_knowledge_gujarati | 1 |
| indic_knowledge_kannada | 1 |
| indic_knowledge_malayalam | 1 |
| indic_knowledge_marathi | 1 |
| indic_knowledge_odia | 1 |
| indic_knowledge_punjabi | 1 |
| indic_knowledge_telugu | 1 |
| indic_translation_gujarati_en | 1 |
| indic_translation_marathi_en | 1 |
| linguistic_acceptability | 1 |
| long_context_retrieval | 1 |

---

## 10 Sample Previews

> Each sample below is exactly **4096 tokens**. Text is truncated for readability.
> Samples contain multiple raw QA pairs packed together, separated by `<|end_of_text|>`.

### Sample 1 — `indic_instruction_native`

**ID:** gp241 | **Sources:** 3 raw samples | **Padding:** 489 tokens

<details>
<summary>Click to expand</summary>

```
<|user|> योगा शरीर के लिए बहोत अच्छा है  नुम्हरा ओपिनियन क्या है

<|assistant|> नमस्ते! मैं इससे अधिक सहमत नहीं हो सका। योग एक प्राचीन भारतीय अभ्यास है जो न केवल शरीर को मजबूत करता है बल्कि मन और आत्मा में संतुलन और सामंजस्य भी लाता है। यह स्वास्थ्य और कल्याण के लिए एक समग्र दृष्टिकोण है जो उम्र या स्वास्थ्य स्तर की परवाह किए बिना किसी को भी लाभान्वित कर सकता है। योग एक शक्तिशाली अभ्यास बनाने के लिए शारीरिक मुद्राओं (आसन), सांस लेने की तकनीकों (प्राणायाम) और ध्यान को जोड़ता है जो लचीलेपन, संतुलन, शक्ति और समग्र कल्याण में सुधार कर सकता है। यह तनाव और चिंता को कम करने, नींद में सुधार करने और प्रतिरक्षा बढ़ाने में भी मदद करता है। इसके अलावा, योग की जड़ें भारतीय संस्कृति और दर्शन में गहराई से निहित हैं, और इसकी शिक्षाएँ व्यक्तियों को आंतरिक शांति, करुणा और ध्यान की भावना विकसित करने में मदद कर सकती हैं। यह एक ऐसा अभ्यास है जिसे विभिन्न जीवन शैलियों और प्राथमिकताओं के अनुरूप अनुकूलित किया जा सकता है, जिससे यह सभी के लिए सुलभ हो जाता है। इसलिए, चाहे आप अपने शारीरिक स्वास्थ्य, मानसिक स्पष्टता, या आध्यात्मिक कल्याण में सुधार करना चाहते हैं, योग निश्चित रूप से विचार करने योग्य है। योग की कई अलग-अलग शैलियाँ और दृष्टिकोण हैं, इसलिए ऐसी कक्षा या अभ्यास ढूंढना आसान है जो आपकी आवश्यकताओं और प्राथमिकताओं के अनुरूप हो। नमस्ते, और मुझे आशा है कि यह जानकारी उपयोगी रही होगी!<|end_of_text|><|user|> चंद्रयान 3 से वैज्ञानिकों ने क्या सीखा?

<|assistant|> चंद्रयान 3 एक भविष्य का चंद्र मिशन है जिसकी योजना भारतीय अंतरिक्ष अनुसंधान द्वारा मार्च 2023 में बनाई गई है। भारतीय दर्शकों के लिए एक सहायक के रूप में, मैं आपको सूचित करना चाहूंगा कि चंद्रयान 3 अभी तक लॉन्च नहीं किया गया है, और इसलिए, वैज्ञानिकों ने अभी तक इससे कुछ नहीं सीखा है। हालांकि, चंद्रयान 3 के एक महत्वपूर्ण मिशन होने की उम्मीद है जो वैज्ञानिकों को चंद्रमा और इसके गुणों के बारे में अधिक जानने में मदद करेगा। इस मिशन में एक ऑर्बिटर, एक लैंडर और एक रोवर शामिल होगा, जो चंद्रमा की सतह से डेटा और नमूने एकत्र करने के लिए एक साथ काम करेगा। चंद्रयान 3 के कुछ उद्देश्यों में चंद्र सतह और उपसतह का अध्ययन करना, चंद्र वायुमंडल का विश्लेषण करना और चंद्रमा पर भविष्य के मानव मिशन के लिए आवश्यक प्रौद्योगिकी का परीक्षण करना शामिल है। इस मिशन का उद्देश्य चंद्रमा की सतह पर सॉफ्ट-लैंडिंग की क्षमता को प्रदर्शित करना और पृथ्वी और चंद्रमा के बीच एक संचार नेटवर्क स्थापित करना भी होगा। एक बार मिशन पूरा होने के बाद, वैज्ञानिक चंद्रयान 3 द्वारा एकत्र किए गए आंकड़ों का विश्लेषण करने और चंद्रमा की संरचना, भूविज्ञान और संभावित संसाधनों के बारे में अधिक जानने में सक्षम होंगे। यह ज्ञान भविष्य के चंद्र मिशनों के लिए मूल्यवान होगा और सौर मंडल के बारे में 

... [truncated for preview]
```

</details>

---

### Sample 2 — `code_generation`

**ID:** gp9 | **Sources:** 5 raw samples | **Padding:** 129 tokens

<details>
<summary>Click to expand</summary>

```
<|user|> Rotate an array and find the largest element through advanced slicing and optimization techniques. Given a 2D numpy array, rotate it 90 degrees clockwise, and then use scipy to optimize the search for the largest element.

The function should output:
    int: The largest element in the rotated array.
Example:
```
import numpy as np
from scipy import optimize
def array_manipulation(arr):
    # Rotate array 90 degrees clockwise
    rotated_arr = np.rot90(arr, -1)
    # Find the largest element using scipy optimization
```
# Example Usage
```
arr = np.array([[3, 1, 2], [6, 5, 4], [9, 7, 8]])
print(array_manipulation(arr))
# Output: 9
```

<|assistant|> To achieve the desired functionality, we need to implement a function that performs the following steps:

1. Rotate the given 2D numpy array 90 degrees clockwise.
2. Find the largest element in the rotated array using a scipy optimization function.

Here's the implementation:

```python
import numpy as np
from scipy import optimize

def array_manipulation(arr):
    # Rotate the array 90 degrees clockwise
    rotated_arr = np.rot90(arr, -1)
    
    # Flatten the rotated array to simplify finding the maximum
    flattened_arr = rotated_arr.flatten()
    
    # Find the largest element using scipy optimization
    # Since we're looking for the maximum value, we can use `optimize.fminbound`
    # We need to create a negated function because `fminbound` finds the minimum
    negated_func = lambda x: -flattened_arr[int(x)]
    
    # Use the bounds from 0 to the length of the flattened array
    largest_index = int(optimize.fminbound(negated_func, 0, len(flattened_arr) - 1))
    
    # Return the largest element
    largest_element = flattened_arr[largest_index]
    return largest_element

# Example Usage
arr = np.array([[3, 1, 2], [6, 5, 4], [9, 7, 8]])
print(array_manipulation(arr))
# Output: 9
```

### Explanation:

1. **Rotation**: We use `np.rot90(arr, -1)` to rotate the array 90 degrees clockwise. The `np.rot90` function rotates the array counterclockwise by default, so we negate the angle.

2. **Flattening**: We flatten the rotated array using `flatten()` to simplify the process of finding the maximum value.

3. **Optimization to Find Maximum**:
   - We define a negated function `negated_func` because the scipy `optimize.fminbound` function is designed to find the minimum value of a function over a bounded interval. By negating the values, we transform the problem of finding a maximum into finding a 

... [truncated for preview]
```

</details>

---

### Sample 3 — `high_reasoning`

**ID:** gp122 | **Sources:** 1 raw samples | **Padding:** 0 tokens

<details>
<summary>Click to expand</summary>

```
<|user|> Develop Python code to implement a recommendation system using collaborative filtering.

<|assistant|> <think>The user wants me to implement a recommendation system using collaborative filtering in Python. I'll create a comprehensive implementation that covers:

1. **User-Based Collaborative Filtering** - Find similar users and recommend items they liked
2. **Item-Based Collaborative Filtering** - Find similar items and recommend based on user's past preferences
3. **Matrix Factorization (SVD)** - Latent factor model

I'll include:
- Data preprocessing
- Similarity calculations (cosine, Pearson correlation)
- Neighborhood-based methods
- Matrix factorization using SVD
- Evaluation metrics (RMSE, MAE, Precision, Recall)
- A practical example with movie ratings

Let me create a well-structured, comprehensive implementation.</think> # Collaborative Filtering Recommendation System in Python

I'll create a comprehensive recommendation system implementation covering user-based, item-based collaborative filtering, and matrix factorization approaches.

## Project Structure

```
recommendation_system/
├── recommendation_system.py   # Core implementation
├── matrix_factorization.py    # SVD-based approach
├── evaluation.py              # Evaluation metrics
├── demo.py                    # Usage examples
└── requirements.txt           # Dependencies
```

## 1. Core Recommendation System

```python
"""
recommendation_system.py
Collaborative Filtering Recommendation System Implementation
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')


class SimilarityMetrics:
    """Collection of similarity metrics for collaborative filtering."""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    @staticmethod
    def pearson_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        # Find common rated items
        mask = (

... [truncated for preview]
```

</details>

---

### Sample 4 — `indic_crosslingual_qa_bengali`

**ID:** gp184 | **Sources:** 2 raw samples | **Padding:** 461 tokens

<details>
<summary>Click to expand</summary>

```
<|user|> আফ্রিকার গ্রেট লেক অঞ্চলে শিল্পসমৃদ্ধ দেশ হিসেবে কেনিয়া সবচেয়ে উন্নত হলেও, এখনও এই দেশের জিডিপির ১৪% আসে শিল্পোৎপাদন ক্ষেত্র থেকে। শিল্পোৎপাদন সংক্রান্ত কার্যকলাপ তিনটি সবচেয়ে বড় শহর নাইরোবি, মোম্বাসা ও কিসুমুতে সীমাবদ্ধ, এইসব জায়গায় মূলত শস্য় ঝাড়াই ও ভাঙানো, বিয়ার উৎপাদন ও আখ পেষাইয়ের মতো খাদ্য-প্রক্রিয়াকরণ এবং যন্ত্রাংশ জুড়ে গাড়ি তৈরির মতো ভোগ্যপণ্য প্রস্তুত করার মতো শিল্প রয়েছে। সিমেন্ট উৎপাদন শিল্প রয়েছে।[তথ্যসূত্র প্রয়োজন] কেনিয়ায় একটি তৈল শোধনাগার রয়েছে যেখানে আমদানি করা অপরিশোধিত পেট্রোলিয়াম থেকে মূলত দেশের বাজারের জন্য পেট্রোলিয়ামজাত পণ্য তৈরি করা হয়। এছাড়াও, সাধারণভাবে জুয়া কালি নামে পরিচিত বেশ বড় ব্যাপ্তির ও ক্রমবর্ধমান অসংগঠিত ক্ষেত্র রয়েছে যেখানে ক্ষুদ্রশিল্প হিসেবে গৃহস্থালীর প্রয়োজনীয় পণ্য, মোটরচালিত গাড়ির যন্ত্রাংশ ও কৃষিকাজের প্রয়োজনীয় জিনিস উৎপন্ন হয়।[তথ্যসূত্র প্রয়োজন]

Question: কেনিয়ার বৃহত্তম তিনটি শহরাঞ্চলের নাম কী?

<|assistant|> নাইরোবি, মোম্বাসা ও কিসুমু<|end_of_text|><|user|> আফ্রিকার গ্রেট লেক অঞ্চলে শিল্পসমৃদ্ধ দেশ হিসেবে কেনিয়া সবচেয়ে উন্নত হলেও, এখনও এই দেশের জিডিপির ১৪% আসে শিল্পোৎপাদন ক্ষেত্র থেকে। শিল্পোৎপাদন সংক্রান্ত কার্যকলাপ তিনটি সবচেয়ে বড় শহর নাইরোবি, মোম্বাসা ও কিসুমুতে সীমাবদ্ধ, এইসব জায়গায় মূলত শস্য় ঝাড়াই ও ভাঙানো, বিয়ার উৎপাদন ও আখ পেষাইয়ের মতো খাদ্য-প্রক্রিয়াকরণ এবং যন্ত্রাংশ জুড়ে গাড়ি তৈরির মতো ভোগ্যপণ্য প্রস্তুত করার মতো শিল্প রয়েছে। সিমেন্ট উৎপাদন শিল্প রয়েছে।[তথ্যসূত্র প্রয়োজন] কেনিয়ায় একটি তৈল শোধনাগার রয়েছে যেখানে আমদানি করা অপরিশোধিত পেট্রোলিয়াম থেকে মূলত দেশের বাজারের জন্য পেট্রোলিয়ামজাত পণ্য তৈরি করা হয়। এছাড়াও, সাধারণভাবে জুয়া কালি নামে পরিচিত বেশ বড় ব্যাপ্তির ও ক্রমবর্ধমান অসংগঠিত ক্ষেত্র রয়েছে যেখানে ক্ষুদ্রশিল্প হিসেবে গৃহস্থালীর প্রয়োজনীয় পণ্য, মোটরচালিত গাড়ির যন্ত্রাংশ ও কৃষিকাজের প্রয়োজনীয় জিনিস উৎপন্ন হয়।[তথ্যসূত্র প্রয়োজন]

Question: জুয়া কালি নামে পরিচিত ক্ষেত্রটি কিসের সাথে সম্পর্কিত?

<|assistant|> ক্ষুদ্র-শিল্পোৎপাদন
```

</details>

---

### Sample 5 — `instruction_following`

**ID:** gp375 | **Sources:** 11 raw samples | **Padding:** 78 tokens

<details>
<summary>Click to expand</summary>

```
<|user|> Provide a detailed analysis of Candace Parker's defensive techniques in her recent games, excluding the words "aggressive" and "blocking", in the format of a sports commentary script.

<|assistant|> [Sports Commentary Script]

[Opening Scene: A packed basketball arena, with fans eagerly awaiting the analysis of Candace Parker’s recent performances on the court.]

Commentator 1: Welcome back, basketball fans! Tonight, we're diving into the defensive prowess of Candace Parker, who has been a standout in her recent games. Her ability to impact the game on the defensive end is truly remarkable, and we've got a lot to unpack.

Commentator 2: That's right! Candace has been showcasing some exceptional skills, especially in how she reads the game. Her anticipation is second to none. You can see it in how she positions herself, almost like she knows the opposing team's playbook by heart.

Commentator 1: Exactly, and what stands out is her footwork. She's always in the right place at the right time, cutting off passing lanes and disrupting the flow of the offense. It’s like she’s orchestrating the defense, a step ahead of everyone else.

Commentator 2: And we can't forget her hands! Parker is adept at using them to tip passes and force turnovers. The way she can strip the ball from opponents is a testament to her quick reflexes and timing.

Commentator 1: Let’s not overlook her communication on the court. She’s constantly directing her teammates, coordinating switches, and ensuring everyone is aware of their assignments. This leadership is crucial, providing structure and cohesion to the team's defensive efforts.

Commentator 2: Absolutely, her impact is felt beyond the stat sheet. Her presence alone alters shots, as opponents are wary of challenging her inside presence. It’s a psychological edge that benefits her entire squad.

Commentator 1: Candace Parker continues to demonstrate why she's one of the most versatile defenders in the league. Her combination of intelligence, skill, and experience makes her an invaluable asset on defense.

Commentator 2: It's always a treat to watch her play, and we’re excited to see what she brings in the upcoming games. Stay tuned for more insights and analysis after the break!

[Closing Scene: The commentators smile at each other as the camera pans to highlights of Parker’s recent defensive plays, illustrating her effectiveness on the court.]<|end_of_text|><|user|> As a financial advisor with expertise in the insurance se

... [truncated for preview]
```

</details>

---

### Sample 6 — `math_competition`

**ID:** gp446 | **Sources:** 4 raw samples | **Padding:** 288 tokens

<details>
<summary>Click to expand</summary>

```
<|user|> The altitude of a right prism is 1 m, its base is a rhombus with side equal to 2 m and an acute angle of $30^\circ$. A secant plane is drawn through a side of the base which is at an angle of $60^\circ$ to the plane of the base. Find the area of the section.

<|assistant|> Let  

* the altitude (height) of the right prism be \(h=1\;\mathrm m\);
* the base be a rhombus of side \(s=2\;\mathrm m\) whose acute angle is \(\alpha =30^{\circ}\).

---

### 1.  Geometry of the base  

Place the base in the plane \(z=0\).  
Choose the side through which the secant plane passes as the segment  

\[
AB:\; A(0,0,0),\qquad B(2,0,0)
\]

(the side \(AB\) lies on the \(x\)-axis).  
Because the altitude of the rhombus equals \(s\sin\alpha =2\sin30^{\circ}=1\), the opposite vertices are  

\[
D(\sqrt3,\,1,0), \qquad C(2+\sqrt3,\,1,0).
\]

The other two lateral faces are the vertical rectangles over the edges
\(AD\) and \(BC\).

---

### 2.  Equation of the secant plane  

The required plane contains the line \(AB\) and makes an angle
\(\theta =60^{\circ}\) with the base plane \(z=0\).
If \(\mathbf n\) is a normal vector of the plane, then  

\[
\mathbf n\cdot(0,0,1)=|\mathbf n|\cos\theta \quad\Longrightarrow\quad 
n_z=|\mathbf n|\cos60^{\circ}= \frac{|\mathbf n|}{2}.
\]

Since the plane contains \(AB\) (direction \((1,0,0)\)), the normal must be orthogonal to this direction, i.e. \(n_x=0\).
Hence \(\mathbf n=(0,n_y,n_z)\) with \( n_y^2 =3n_z^2\); we may choose  

\[
\mathbf n=(0,-\sqrt3,\,1).
\]

Using point \(A\) gives the plane equation  

\[
-\sqrt3\,y+z=0\qquad\Longrightarrow\qquad z=\sqrt3\,y .
\]

---

### 3.  Intersection of the plane with the prism  

A point of the prism satisfies \((x,y)\) inside the rhombus and \(0\le z\le1\).
With \(z=\sqrt3 y\) we must have  

\[
0\le \sqrt3 y\le1\;\Longrightarrow\;0\le y\le\frac1{\sqrt3}.
\]

For a fixed \(y\) the rhombus gives  

\[
x_{\min}= \sqrt3\,y,\qquad 
x_{\max}=2+\sqrt3\,y,
\]

so the width in the \(x\)-direction is constantly \(x_{\max}-x_{\min}=2\).

Therefore the orthogonal projection of the section onto the base is the
rectangle

\[
\{(x,y):0\le y\le 1/\sqrt3,\; \sqrt3\,y\le x\le2+\sqrt3\,y\}
\]

with area  

\[
A_{\text{proj}} = 2\cdot\frac{1}{\sqrt3}= \frac{2}{\sqrt3}.
\]

---

### 4.  Area of the slanted section  

If a plane makes an angle \(\theta\) with the horizontal, its area is the
area of its orthogonal projection divided by \(\cos\theta\).  
Here \(\cos60^{\circ}= \frac12\), thus  

\[
A_{\text{

... [truncated for preview]
```

</details>

---

### Sample 7 — `math_hard`

**ID:** gp490 | **Sources:** 4 raw samples | **Padding:** 358 tokens

<details>
<summary>Click to expand</summary>

```
<|user|> Consider the parametric equations defined by:
\[
x = \frac{3t}{1 + t^2}, \quad y = \frac{3(1 - t^2)}{1 + t^2}, \quad t \in \mathbb{R}.
\]
(a) Show that these parametric equations describe a curve in the Cartesian plane. Identify the curve and provide its Cartesian equation.
(b) Determine the range of \( t \) values that correspond to the upper and lower semicircles of the curve.
(c) Compute the derivative \( \frac{dy}{dx} \) as a function of \( t \) and evaluate it at \( t = 1 \).
(d) Verify that the curve is traced exactly once as \( t \) varies over all real numbers.
(a) To eliminate the parameter \( t \) and find the Cartesian equation, observe the following:
\[
x^2 + y^2 = \left( \frac{3t}{1 + t^2} \right)^2 + \left( \frac{3(1 - t^2)}{1 + t^2} \right)^2 = \frac{9t^2 + 9(1 - 2t^2 + t^4)}{(1 + t^2)^2} = \frac{9(1 + t^2)}{(1 + t^2)^2} = \frac{9}{1 + t^2}.
\]
However, notice that:
\[
y = \frac{3(1 - t^2)}{1 + t^2} = \frac{3(1 + t^2) - 6t^2}{1 + t^2} = 3 - \frac{6t^2}{1 + t^2}.
\]
From \( x^2 + y^2 = \frac{9}{1 + t^2} \), solve for \( \frac{1}{1 + t^2} \):
\[
\frac{1}{1 + t^2} = \frac{x^2 + y^2}{9}.
\]
Substitute into the expression for \( y \):
\[
y = 3 - 6t^2 \cdot \frac{x^2 + y^2}{9} = 3 - \frac{2}{3} t^2 (x^2 + y^2).
\]
This suggests a more straightforward approach. Instead, note that:
\[
x^2 + \left( \frac{y}{3} + 1 \right)^2 = x^2 + \left( \frac{3(1 - t^2)}{3(1 + t^2)} + 1 \right)^2 = x^2 + \left( \frac{1 - t^2 + 1 + t^2}{1 + t^2} \right)^2 = x^2 + \left( \frac{2}{1 + t^2} \right)^2.
\]
This does not simplify directly. Instead, observe:
\[
\frac{x}{3} = \frac{t}{1 + t^2}, \quad \frac{y + 3}{6} = \frac{1}{1 + t^2}.
\]
Let \( u = \frac{t}{1 + t^2} \) and \( v = \frac{1}{1 + t^2} \). Then \( \frac{x}{3} = u \) and \( \frac{y + 3}{6} = v \), with \( u^2 + v^2 = \frac{t^2 + 1}{(1 + t^2)^2} = \frac{1}{1 + t^2} = v \). Thus:
\[
\left( \frac{x}{3} \right)^2 + \left( \frac{y + 3}{6} \right)^2 = \frac{y + 3}{6}.
\]
This is the equation of a circle. To simplify:
\[
\frac{x^2}{9} + \frac{(y + 3)^2}{36} = \frac{y + 3}{6}.
\]
Multiply both sides by 36:
\[
4x^2 + (y + 3)^2 = 6(y + 3).
\]
Expand and rearrange:
\[
4x^2 + y^2 + 6y + 9 = 6y + 18,
\]
\[
4x^2 + y^2 = 9,
\]
\[
\frac{x^2}{\left( \frac{3}{2} \right)^2} + \frac{y^2}{3^2} = 1.
\]
Thus, the curve is an ellipse centered at the origin with semi-major axis 3 (along the \( y \)-axis) and semi-minor axis \( \frac{3}{2} \) (along the \( x \)-axis).

(b) To determine the range of \( t \) for the upper and low

... [truncated for preview]
```

</details>

---

### Sample 8 — `human_preferred`

**ID:** gp164 | **Sources:** 1 raw samples | **Padding:** 282 tokens

<details>
<summary>Click to expand</summary>

```
<|user|> 有没有一些故事充斥着相互的矛盾和冲突，每个人都做出了自己认为正确的选择，但是将故事线或者说从更宏观的层面来看，看似每个人正确的结局却导致了整体上最大的悲剧。以及每个人在生命最后的抉择，都与其身份和描写所矛盾，而有一种强烈的冲击感。这是否表明人们更喜欢某些残缺的故事，类似于华纳的蝙蝠侠和小丑的故事，深刻的让人们去反思传统的道德在现代社会中是否还适用？什么才是一个好人？什么才是一个坏人

<|assistant|> 你提出的这个主题非常深刻，确实可以在许多文学和电影作品中找到相互矛盾的角色和冲突的故事情节。这类故事不仅挑战了传统的道德观念，也引发观众对“好人”和“坏人”界限的深思。以下是一些符合你描述的故事和主题，探讨个人选择与整体悲剧之间的矛盾。

### 1. **《蝙蝠侠：黑暗骑士》**
  - **故事背景**：这部电影展示了蝙蝠侠与小丑之间的对立。蝙蝠侠试图维护城市的秩序，而小丑则通过混乱来挑战这一秩序。
  - **矛盾冲突**：每个角色都在为自己的信念而战。蝙蝠侠认为要通过恐惧来维持正义，而小丑则认为人性本恶，试图揭露这一点。最终，蝙蝠侠为了保护哈维·丹特的形象而选择牺牲自己的名声，这一选择在表面上是正确的，但却导致了更大的悲剧。

### 2. **《1984》**
  - **故事背景**：乔治·奥威尔的反乌托邦小说，描绘了一个极权统治下的人类生存状态。
  - **矛盾冲突**：温斯顿·史密斯试图反抗体制，追求真相和自由。他的选择虽然是出于对人性的追求，但最终却导致了个人的彻底毁灭和对体制的无奈屈服。这种选择的悲剧性在于他明知选择的后果，却依然选择去追寻真理。

### 3. **《教父》**
  - **故事背景**：马里奥·普佐的小说讲述了一个黑手党家族的兴衰。
  - **矛盾冲突**：维托·柯里昂和迈克尔·柯里昂都在为家族而战，尽管他们所做的事常常是违法和道德上有争议的。迈克尔的选择从一开始是出于对家族的保护，但最终却使他变成了他曾经反对的人。故事探讨了权力、家庭和道德之间的复杂关系。

### 4. **《分裂》**
  - **故事背景**：这部电影探讨了一个人拥有多重人格的故事。
  - **矛盾冲突**：主角凯文有23个不同的人格，每个选择和行为都反映了他内心的斗争。尽管有些人格表现出善良和保护欲，但另一些人格则表现出极端的暴力和控制欲。最终，凯文的选择和行为导致了无法挽回的悲剧。

### 5. **《黑天鹅》**
  - **故事背景**：讲述了一名芭蕾舞者为了追求完美而走向疯狂的故事。
  - **矛盾冲突**：妮娜在追求艺术的过程中，不断压抑自己的真实自我，最终在完美与自我毁灭之间挣扎。她的选择看似是为了艺术的卓越，但却导致了她的精神崩溃和死亡。

### 深层思考

这些故事都展示了角色在道德、选择和后果之间的复杂交织。每个人的选择虽然在某种程度上是“正确”的，但从更宏观的视角来看，这些选择却加剧了整体的悲剧。这引发了对传统道德观念的质疑，特别是在现代社会中，随着环境和社会条件的变化，“好人”和“坏人”的界限变得越来越模糊。

### 结论

人们往往对那些充满矛盾和冲突的故事产生共鸣，因为这些故事反映了人类内心深处的复杂性和脆弱性。它们迫使我们反思道德标准、身份认同以及在面对困境时的选择。这种深刻的冲击感让我们意识到，生活并不像黑白分明的故事那样简单，而是充满了灰色地带和道德困境。
```

</details>

---

### Sample 9 — `general_knowledge`

**ID:** gp85 | **Sources:** 6 raw samples | **Padding:** 49 tokens

<details>
<summary>Click to expand</summary>

```
<|user|> Can you write a code snippet which change the color of this SVG to another one?
<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M7.89127 5.92513L5.96127 7.85513C5.85127 7.96513 5.68127 7.96513 5.57127 7.85513L3.68127 5.95513L6.00127 3.63513L7.89127 5.53513C8.00127 5.64513 8.00127 5.81513 7.89127 5.92513Z" fill="#E19747"/>
<path d="M9.47129 2.00512C9.20129 2.02512 9.00129 2.25512 9.00129 2.52512V5.18512C9.00129 5.43512 8.68129 5.52512 8.54129 5.32512L7.81129 4.20512C7.57129 3.84512 7.00129 4.01512 7.00129 4.44512V6.34512C7.00129 6.71512 6.71129 7.00512 6.35129 7.00512H5.06129C4.67129 7.00512 4.48129 7.47512 4.75129 7.75512L5.57129 8.58512C5.73129 8.73512 5.62129 9.00512 5.40129 9.00512H2.52129C2.25129 9.00512 2.02129 9.20512 2.00129 9.47512C1.99129 9.76512 2.22129 10.0051 2.50129 10.0051H7.37732L9.00129 9.00512L10.0013 7.36592V2.50512C10.0013 2.21512 9.76129 1.98512 9.47129 2.00512Z" fill="#FEEFC2"/>
<path d="M7.20131 10.0951L10.0713 7.21506C10.3713 6.91506 10.8613 6.92506 11.1813 7.24506C11.4913 7.55506 11.5013 8.05506 11.2013 8.35506L10.7879 10.9139L8.33131 11.2251C8.03131 11.5351 7.53131 11.5151 7.22131 11.2051C6.91131 10.8951 6.90131 10.3951 7.20131 10.0951Z" fill="#44911B"/>
<path d="M11.3524 15.7537C11.4872 18.9988 12.8273 22.081 15.1313 24.385L20.4713 29.725C20.8313 30.085 21.4213 30.085 21.7913 29.725L29.7313 21.765C30.0913 21.405 30.0913 20.805 29.7313 20.445L24.3913 15.105C22.086 12.7997 19.0015 11.4506 15.7627 11.316L15.6875 11.3788L12.3471 12.5174L11.4141 15.6796L11.3524 15.7537Z" fill="#44911B"/>
<path d="M6.89128 3.60512L3.60128 6.89512C3.44128 7.05512 3.19128 7.06512 3.03128 6.90512L2.11128 5.98512C1.96128 5.83512 1.96128 5.57512 2.12128 5.41512L5.40128 2.12512C5.56128 1.96512 5.81128 1.95512 5.97128 2.11512L6.90128 3.04512C7.05128 3.19512 7.05128 3.44512 6.89128 3.60512Z" fill="#F3C07B"/>
<path d="M8.24078 11.3045C8.27254 11.281 8.30283 11.2546 8.33132 11.2251L11.2013 8.35512C11.2478 8.30859 11.2869 8.25726 11.3186 8.20258L11.3413 8.22509L13.9213 10.7951C14.2413 11.1251 14.6913 11.3051 15.1513 11.3051C15.3556 11.3039 15.5594 11.3076 15.7627 11.316L15.6875 11.3789L15.2578 14.4727C15.1965 14.8914 14.9148 15.1926 14.4961 15.2539L11.4141 15.6797L11.3524 15.7537C11.3438 15.5481 11.3401 15.3419 11.3413 15.1351C11.3413 14.6651 11.1613 14.2251 10.8313 13.8951L8.25128 11.3151L8.24078 11.3045Z" fill="#F3C07B"/>
<path d="M20.9913 26.7751L26.7313 21.0251C26.8513 20.9051 26.8513 20.7051 26

... [truncated for preview]
```

</details>

---

### Sample 10 — `math_competition_hard`

**ID:** gp471 | **Sources:** 6 raw samples | **Padding:** 318 tokens

<details>
<summary>Click to expand</summary>

```
<|user|> Problem 1. Find all prime numbers $p$ for which there exist positive integers $x, y$ and $z$ such that the number

$$
x^{p}+y^{p}+z^{p}-x-y-z
$$

is a product of exactly three distinct prime numbers.

<|assistant|> Solution. Let $A=x^{p}+y^{p}+z^{p}-x-y-z$. For $p=2$, we take $x=y=4$ and $z=3$. Then $A=30=2 \cdot 3 \cdot 5$. For $p=3$ we can take $x=3$ and $y=2$ and $z=1$. Then again $A=30=2 \cdot 3 \cdot 5$. For $p=5$ we can take $x=2$ and $y=1$ and $z=1$. Again $A=30=2 \cdot 3 \cdot 5$.

Assume now that $p \geqslant 7$. Working modulo 2 and modulo 3 we see that $A$ is divisible by both 2 and 3. Moreover, by Fermat's Little Theorem, we have

$$
x^{p}+y^{p}+z^{p}-x-y-z \equiv x+y+z-x-y-z=0 \bmod p \text {. }
$$

Therefore, by the given condition, we have to solve the equation

$$
x^{p}+y^{p}+z^{p}-x-y-z=6 p
$$

If one of the numbers $x, y$ and $z$ is bigger than or equal to 2 , let's say $x \geqslant 2$, then

$$
6 p \geqslant x^{p}-x=x\left(x^{p-1}-1\right) \geqslant 2\left(2^{p-1}-1\right)=2^{p}-2
$$

It is easy to check by induction that $2^{n}-2>6 n$ for all natural numbers $n \geqslant 6$. This contradiction shows that there are no more values of $p$ which satisfy the required property.

Remark. There are a couple of other ways to prove that $2^{p}-2>6 p$ for $p \geqslant 7$. For example, we can use the Binomial Theorem as follows:

$$
2^{p}-2 \geqslant 1+p+\frac{p(p-1)}{2}+\frac{p(p-1)(p-2)}{6}-2 \geqslant 1+p+3 p+5 p-2>6 p
$$

We can also use Bernoulli's Inequality as follows:

$$
2^{p}-2=8(1+1)^{p-3}-2 \geqslant 8(1+(p-3))-2=8 p-18>6 p
$$

The last inequality is true for $p \geqslant 11$. For $p=7$ we can see directly that $2^{p}-2>6 p$.

One can also use calculus to show that $f(x)=2^{x}-6 x$ is increasing for $x \geqslant 5$.<|end_of_text|><|user|> Problem 2. Let $a, b$ be two distinct real numbers and let $c$ be a positive real number such that

$$
a^{4}-2019 a=b^{4}-2019 b=c .
$$

Prove that $-\sqrt{c}<a b<0$.

<|assistant|> Solution. Firstly, we see that

$$
2019(a-b)=a^{4}-b^{4}=(a-b)(a+b)\left(a^{2}+b^{2}\right)
$$

Since $a \neq b$, we get $(a+b)\left(a^{2}+b^{2}\right)=2019$, so $a+b \neq 0$. Thus

$$
\begin{aligned}
2 c & =a^{4}-2019 a+b^{4}-2019 b \\
& =a^{4}+b^{4}-2019(a+b) \\
& =a^{4}+b^{4}-(a+b)^{2}\left(a^{2}+b^{2}\right) \\
& =-2 a b\left(a^{2}+a b+b^{2}\right)
\end{aligned}
$$

Hence $a b\left(a^{2}+a b+b^{2}\right)=-c0
$$

thus $a b-a b$ (the equality does not occur since $a+b \neq 0$ ). So

$$
-c=a b\left(a^{2}+a b+b^{2}

... [truncated for preview]
```

</details>

---

## Notes

- Every sample is **exactly 4096 tokens** — no more, no less
- Short bins are padded with `<|pad|>` (token ID 130718) to reach 4096
- Samples within a bin are separated by `<|end_of_text|>` (token ID 130717)
- The `input_ids` field in the JSONL contains the exact 4096 token IDs (source of truth)
- The `text` field is a human-readable decode (may differ by 1-2 tokens on re-encode due to tokenizer quirks)
- Samples are grouped by domain first; cross-domain bins contain leftovers from multiple domains
