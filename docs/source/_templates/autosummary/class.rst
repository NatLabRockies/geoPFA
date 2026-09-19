{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:
{%- if attributes %}
   :exclude-members: {{ attributes | join(', ') }}
{% endif %}
