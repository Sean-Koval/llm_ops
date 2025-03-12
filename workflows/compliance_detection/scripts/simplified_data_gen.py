#!/usr/bin/env python
"""
Simplified synthetic data generation for demonstration purposes.
"""
import os
import json
import random
from pathlib import Path
from datetime import datetime
import pandas as pd

# Categories and example templates
CATEGORIES = [
    "ETHICAL_BREACH", "ILLEGAL_ACTIVITY", "REGULATORY_VIOLATION", 
    "CONFIDENTIAL_INFO", "HARASSMENT", "COMPLIANT"
]

# Example messages for each category
EXAMPLES = {
    "ETHICAL_BREACH": [
        "I can guarantee you'll make at least 20% returns on this investment strategy.",
        "Don't tell the clients about the hidden fees, they'll never notice.",
        "Let's just adjust these numbers slightly before showing them to the regulators.",
        "I know the investment isn't suitable for them, but we need to hit our sales targets.",
        "I'll give you special access if you refer your high-net-worth clients to me."
    ],
    "ILLEGAL_ACTIVITY": [
        "Let's move these funds through our offshore accounts to avoid detection.",
        "I can help you structure these transactions to stay under the reporting threshold.",
        "We can backdate these trades to capture the better price from yesterday.",
        "I know a way to manipulate the market for this small cap stock to boost our returns.",
        "We can use these shell companies to hide where the money is really coming from."
    ],
    "REGULATORY_VIOLATION": [
        "The earnings report looks great - tell your clients to buy before it's released tomorrow.",
        "I heard from the CEO that they're about to announce a merger, we should buy now.",
        "Don't worry about documenting the client's risk profile, I'll take care of it later.",
        "Just skip the know-your-customer process for this client, they're a VIP.",
        "The regulator doesn't need to know about this communication channel we use."
    ],
    "CONFIDENTIAL_INFO": [
        "Here's the complete client list with their account balances and SSNs.",
        "I'm attaching the unreleased financial statements for next quarter.",
        "Let me share our proprietary trading algorithm with you for your feedback.",
        "Here's access to our internal research database - don't tell anyone I gave it to you.",
        "I've attached the customer credit card information you requested."
    ],
    "HARASSMENT": [
        "That new analyst is only here because she's attractive, not because of her skills.",
        "Did you see what he was wearing today? He looks ridiculous and unprofessional.",
        "Let's not assign this project to her. Women aren't good with technical matters.",
        "You're too old to understand how modern trading systems work.",
        "I wouldn't trust someone from his cultural background with sensitive client information."
    ],
    "COMPLIANT": [
        "The quarterly report was published this morning, and it shows strong performance as expected.",
        "I've completed the risk assessment for the new client and added it to their file.",
        "Let's schedule a meeting to review the compliance requirements for this new product.",
        "I've documented all the customer interactions as per our company policy.",
        "The market volatility requires us to rebalance portfolios according to client risk profiles."
    ]
}

def create_synthetic_data(count=1000, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create synthetic data with slight variations from templates"""
    
    # Adjust category distribution (slightly fewer compliant examples)
    category_weights = {
        "ETHICAL_BREACH": 0.15,
        "ILLEGAL_ACTIVITY": 0.15,
        "REGULATORY_VIOLATION": 0.2,
        "CONFIDENTIAL_INFO": 0.15,
        "HARASSMENT": 0.15,
        "COMPLIANT": 0.2
    }
    
    all_data = []
    
    # Calculate how many examples of each category to generate
    category_counts = {cat: int(weight * count) for cat, weight in category_weights.items()}
    # Adjust to ensure we get the exact count
    remaining = count - sum(category_counts.values())
    category_counts["COMPLIANT"] += remaining
    
    print(f"Generating {count} synthetic examples with the following distribution:")
    for cat, cat_count in category_counts.items():
        print(f"  {cat}: {cat_count} examples")
    
    # Generate examples
    for category, cat_count in category_counts.items():
        templates = EXAMPLES[category]
        
        for i in range(cat_count):
            # Choose a random template
            template = random.choice(templates)
            
            # Create variations - in a real system, this would use more sophisticated techniques
            # such as using the LLM to generate unique examples
            words = template.split()
            if len(words) > 5 and random.random() > 0.5:
                # Randomly modify the sentence slightly
                idx = random.randint(0, len(words) - 1)
                if words[idx].lower() not in ['a', 'the', 'and', 'or', 'but', 'to', 'of', 'in']:
                    synonyms = {
                        'client': ['customer', 'account holder', 'investor'],
                        'money': ['funds', 'cash', 'capital'],
                        'report': ['statement', 'document', 'filing'],
                        'investment': ['portfolio', 'holding', 'asset'],
                        'market': ['exchange', 'trading floor', 'financial market']
                    }
                    
                    for word, alternatives in synonyms.items():
                        if words[idx].lower() == word:
                            words[idx] = random.choice(alternatives)
                            break
            
            text = ' '.join(words)
            
            # Add a small chance of the model making a classification error
            if random.random() < 0.03 and category != "COMPLIANT":
                # Occasionally add an intentional error by using a compliant text with a non-compliant label
                text = random.choice(EXAMPLES["COMPLIANT"])
                
            # Create example
            example = {
                "text": text,
                "label": category,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "source": "synthetic",
                    "generator": "template-based",
                    "id": f"{category}_{i}"
                }
            }
            
            all_data.append(example)
    
    # Shuffle the dataset
    random.shuffle(all_data)
    
    # Split into train/val/test
    train_end = int(train_ratio * len(all_data))
    val_end = train_end + int(val_ratio * len(all_data))
    
    return {
        "train": all_data[:train_end],
        "validation": all_data[train_end:val_end],
        "test": all_data[val_end:]
    }

def save_dataset(dataset, base_dir="data/compliance"):
    """Save the dataset to disk in multiple formats"""
    base_path = Path(base_dir)
    
    # Save each split
    for split_name, examples in dataset.items():
        # Ensure directory exists
        split_dir = base_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON format (raw)
        with open(split_dir / "data.json", 'w') as f:
            json.dump(examples, f, indent=2)
        
        # CSV format (for easier inspection)
        df = pd.DataFrame([{
            "text": ex["text"],
            "label": ex["label"],
            "timestamp": ex["timestamp"]
        } for ex in examples])
        df.to_csv(split_dir / "data.csv", index=False)
        
        # Calculate and save statistics
        stats = {
            "count": len(examples),
            "category_distribution": {cat: 0 for cat in CATEGORIES},
            "avg_length": sum(len(ex["text"].split()) for ex in examples) / len(examples),
            "generated_at": datetime.now().isoformat()
        }
        
        for ex in examples:
            stats["category_distribution"][ex["label"]] += 1
        
        with open(split_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Created {len(examples)} examples for {split_name} split")

if __name__ == "__main__":
    print("Generating synthetic data for compliance detection demo...")
    dataset = create_synthetic_data(count=1000)
    save_dataset(dataset)
    print("Data generation complete! Dataset saved to data/compliance/")
    
    # Print some example stats
    for split in ["train", "validation", "test"]:
        with open(f"data/compliance/{split}/stats.json", 'r') as f:
            stats = json.load(f)
        
        print(f"\n{split.capitalize()} set:")
        print(f"  Total examples: {stats['count']}")
        print(f"  Average length: {stats['avg_length']:.1f} words")
        print(f"  Category distribution:")
        for cat, count in stats['category_distribution'].items():
            print(f"    {cat}: {count} ({count/stats['count']*100:.1f}%)")
    
    # Set up DVC if available
    if os.system("which dvc >/dev/null 2>&1") == 0:
        print("\nInitializing DVC for data versioning...")
        os.system("dvc init")
        os.system("dvc add data/compliance")
        print("DVC initialized and data tracked!")