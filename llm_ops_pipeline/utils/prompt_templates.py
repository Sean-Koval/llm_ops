"""
Sophisticated prompt templating system for LLM prompts.

This module extends the basic prompt management system with more advanced
templating capabilities, including conditional blocks, few-shot examples,
and template inheritance.
"""

import re
import json
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PromptTemplate:
    """
    Advanced prompt templating system that goes beyond simple variable substitution.
    
    Features:
    - Variable interpolation with formatting options
    - Conditional blocks based on provided variables
    - Loop constructs for generating repetitive content
    - Inclusion of few-shot examples
    - Template inheritance and composition
    """
    
    def __init__(
        self, 
        template: str,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        functions: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize a prompt template.
        
        Args:
            template: The template string with placeholders
            few_shot_examples: Optional list of examples for few-shot learning
            functions: Optional dictionary of functions that can be called in templates
        """
        self.template = template
        self.few_shot_examples = few_shot_examples or []
        self.functions = functions or {}
        
        # Add built-in functions
        self._add_built_in_functions()
    
    def _add_built_in_functions(self):
        """Add built-in template functions."""
        self.functions.update({
            "uppercase": lambda s: str(s).upper(),
            "lowercase": lambda s: str(s).lower(),
            "capitalize": lambda s: str(s).capitalize(),
            "strip": lambda s: str(s).strip(),
            "truncate": lambda s, length=100: str(s)[:int(length)] + "..." if len(str(s)) > int(length) else str(s),
            "join": lambda items, sep=", ": sep.join([str(item) for item in items]),
            "length": lambda items: len(items) if hasattr(items, "__len__") else 0,
        })
    
    def _process_variable(self, var_name: str, context: Dict[str, Any]) -> str:
        """
        Process a variable with possible function calls.
        
        Examples:
            {{variable}} - Simple variable
            {{variable|uppercase}} - Apply uppercase function
            {{variable|truncate:50}} - Truncate to 50 chars
        """
        # Check for function calls
        if "|" in var_name:
            parts = var_name.split("|")
            var_name = parts[0].strip()
            
            # Get the variable value
            value = context.get(var_name, "")
            
            # Apply functions in sequence
            for func_call in parts[1:]:
                func_parts = func_call.strip().split(":")
                func_name = func_parts[0].strip()
                
                if func_name in self.functions:
                    # Get function arguments
                    args = []
                    kwargs = {}
                    
                    if len(func_parts) > 1:
                        for arg in func_parts[1:]:
                            if "=" in arg:
                                k, v = arg.split("=", 1)
                                kwargs[k.strip()] = v.strip()
                            else:
                                args.append(arg.strip())
                    
                    # Apply the function
                    value = self.functions[func_name](value, *args, **kwargs)
                else:
                    logger.warning(f"Unknown function: {func_name}")
            
            return str(value)
        else:
            # Simple variable
            return str(context.get(var_name.strip(), ""))
    
    def _process_conditional(self, condition: str, content: str, context: Dict[str, Any]) -> str:
        """Process conditional blocks like {% if variable %}content{% endif %}."""
        condition = condition.strip()
        
        # Evaluate the condition
        result = False
        
        if " and " in condition:
            subconditions = condition.split(" and ")
            result = all(self._evaluate_condition(sub.strip(), context) for sub in subconditions)
        elif " or " in condition:
            subconditions = condition.split(" or ")
            result = any(self._evaluate_condition(sub.strip(), context) for sub in subconditions)
        else:
            result = self._evaluate_condition(condition, context)
        
        return content if result else ""
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        # Check for negation
        negated = False
        if condition.startswith("not "):
            condition = condition[4:].strip()
            negated = True
        
        # Check for comparisons
        if " == " in condition:
            left, right = condition.split(" == ", 1)
            result = self._get_condition_value(left, context) == self._get_condition_value(right, context)
        elif " != " in condition:
            left, right = condition.split(" != ", 1)
            result = self._get_condition_value(left, context) != self._get_condition_value(right, context)
        elif " > " in condition:
            left, right = condition.split(" > ", 1)
            result = self._get_condition_value(left, context) > self._get_condition_value(right, context)
        elif " < " in condition:
            left, right = condition.split(" < ", 1)
            result = self._get_condition_value(left, context) < self._get_condition_value(right, context)
        elif " in " in condition:
            item, collection = condition.split(" in ", 1)
            item_value = self._get_condition_value(item, context)
            collection_value = self._get_condition_value(collection, context)
            result = item_value in collection_value if hasattr(collection_value, "__contains__") else False
        else:
            # Simple variable check
            result = bool(context.get(condition, False))
        
        return not result if negated else result
    
    def _get_condition_value(self, value_str: str, context: Dict[str, Any]) -> Any:
        """Get a value for condition evaluation."""
        value_str = value_str.strip()
        
        # Check if it's a quoted string
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        # Check if it's a number
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            # It's a variable
            return context.get(value_str, "")
    
    def _process_loop(self, loop_var: str, collection_var: str, content: str, context: Dict[str, Any]) -> str:
        """Process loop blocks like {% for item in items %}content{% endfor %}."""
        collection = context.get(collection_var.strip(), [])
        
        if not collection or not hasattr(collection, "__iter__"):
            return ""
        
        results = []
        loop_var = loop_var.strip()
        
        for i, item in enumerate(collection):
            # Create a new context with the loop variable
            loop_context = context.copy()
            loop_context[loop_var] = item
            loop_context[f"{loop_var}_index"] = i
            loop_context[f"{loop_var}_number"] = i + 1
            loop_context[f"{loop_var}_first"] = i == 0
            loop_context[f"{loop_var}_last"] = i == len(collection) - 1
            
            # Process the content with the new context
            results.append(self._process_template_segment(content, loop_context))
        
        return "".join(results)
    
    def _process_include(self, template_name: str, context: Dict[str, Any]) -> str:
        """Process include blocks like {% include "template_name" %}."""
        # This would typically load a template from a repository
        # For simplicity, we'll assume templates are in context
        if "templates" in context and template_name in context["templates"]:
            template_content = context["templates"][template_name]
            return self._process_template_segment(template_content, context)
        else:
            logger.warning(f"Template not found: {template_name}")
            return ""
    
    def _process_template_segment(self, template: str, context: Dict[str, Any]) -> str:
        """Process a segment of the template with all features."""
        # Process conditions (if/endif blocks)
        condition_pattern = r"{%\s*if\s+([^%]+)\s*%}(.*?){%\s*endif\s*%}"
        while re.search(condition_pattern, template, re.DOTALL):
            template = re.sub(
                condition_pattern,
                lambda m: self._process_conditional(m.group(1), m.group(2), context),
                template,
                flags=re.DOTALL
            )
        
        # Process loops (for/endfor blocks)
        loop_pattern = r"{%\s*for\s+([^\s]+)\s+in\s+([^\s]+)\s*%}(.*?){%\s*endfor\s*%}"
        while re.search(loop_pattern, template, re.DOTALL):
            template = re.sub(
                loop_pattern,
                lambda m: self._process_loop(m.group(1), m.group(2), m.group(3), context),
                template,
                flags=re.DOTALL
            )
        
        # Process includes
        include_pattern = r"{%\s*include\s+[\"']([^\"']+)[\"']\s*%}"
        while re.search(include_pattern, template):
            template = re.sub(
                include_pattern,
                lambda m: self._process_include(m.group(1), context),
                template
            )
        
        # Process variables
        var_pattern = r"{{([^}]+)}}"
        template = re.sub(
            var_pattern,
            lambda m: self._process_variable(m.group(1), context),
            template
        )
        
        return template
    
    def _add_few_shot_examples(self, template: str, context: Dict[str, Any]) -> str:
        """Add few-shot examples to the template if requested."""
        if not self.few_shot_examples:
            return template
        
        # Check if few-shot examples are requested
        if context.get("use_few_shot", True) is False:
            return template
        
        # Format each example
        examples_text = []
        for i, example in enumerate(self.few_shot_examples):
            # Create a context with the example data
            example_context = context.copy()
            example_context.update(example)
            example_context["example_index"] = i
            example_context["example_number"] = i + 1
            
            # Process the example template if provided
            if "example_template" in context:
                example_text = self._process_template_segment(context["example_template"], example_context)
            else:
                # Default formatting if no template provided
                example_text = json.dumps(example, indent=2)
            
            examples_text.append(example_text)
        
        # Inject the examples at the FEW_SHOT_EXAMPLES marker
        if "FEW_SHOT_EXAMPLES" in template:
            examples_str = "\n\n".join(examples_text)
            return template.replace("FEW_SHOT_EXAMPLES", examples_str)
        else:
            # Append to the end if no marker
            return template + "\n\nExamples:\n" + "\n\n".join(examples_text)
    
    def render(self, context: Dict[str, Any]) -> str:
        """
        Render the template with the given context.
        
        Args:
            context: Dictionary of values to use in the template
            
        Returns:
            The rendered template string
        """
        # Process the main template
        result = self._process_template_segment(self.template, context)
        
        # Add few-shot examples if needed
        result = self._add_few_shot_examples(result, context)
        
        return result


class PromptTemplateLibrary:
    """
    Manages a collection of prompt templates with inheritance and composition.
    
    This class allows templates to inherit from other templates and
    provides a centralized way to load and manage templates.
    """
    
    def __init__(self, templates_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the template library.
        
        Args:
            templates_dir: Optional directory where templates are stored
        """
        self.templates = {}
        self.templates_dir = Path(templates_dir) if templates_dir else None
    
    def add_template(self, name: str, template: Union[str, PromptTemplate], 
                    few_shot_examples: Optional[List[Dict[str, Any]]] = None) -> PromptTemplate:
        """
        Add a template to the library.
        
        Args:
            name: The template name
            template: Either a template string or a PromptTemplate instance
            few_shot_examples: Optional list of examples for few-shot learning
            
        Returns:
            The added PromptTemplate
        """
        if isinstance(template, str):
            template = PromptTemplate(template, few_shot_examples)
        
        self.templates[name] = template
        return template
    
    def get_template(self, name: str) -> PromptTemplate:
        """
        Get a template by name.
        
        Args:
            name: The template name
            
        Returns:
            The PromptTemplate instance
            
        Raises:
            ValueError: If the template is not found
        """
        if name in self.templates:
            return self.templates[name]
        
        # Try to load from file if templates_dir is set
        if self.templates_dir:
            template_path = self.templates_dir / f"{name}.j2"
            if template_path.exists():
                with open(template_path, "r") as f:
                    template_str = f.read()
                
                # Check for few-shot examples file
                examples_path = self.templates_dir / f"{name}_examples.json"
                few_shot_examples = None
                if examples_path.exists():
                    with open(examples_path, "r") as f:
                        few_shot_examples = json.load(f)
                
                template = PromptTemplate(template_str, few_shot_examples)
                self.templates[name] = template
                return template
        
        raise ValueError(f"Template not found: {name}")
    
    def render(self, name: str, context: Dict[str, Any]) -> str:
        """
        Render a template by name with the given context.
        
        Args:
            name: The template name
            context: Dictionary of values to use in the template
            
        Returns:
            The rendered template string
        """
        template = self.get_template(name)
        return template.render(context)
    
    def remove_template(self, name: str) -> None:
        """
        Remove a template from the library.
        
        Args:
            name: The template name
        """
        if name in self.templates:
            del self.templates[name]