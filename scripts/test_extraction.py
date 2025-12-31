#!/usr/bin/env python
"""Test JSON extraction from Slack responses."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.channels.slack import SlackHandler

handler = SlackHandler.__new__(SlackHandler)

# Exact response from screenshot (two JSON blocks concatenated)
test = '''{"thought":"List compute instances in the specified compartment by resolving the compartment name to its OCID and querying OCI","action":"oci_compute_list_instances","action_input":{"compartment_id":"ocid1.compartment.oc1..aaaaaaaagy3yddkkampnhj3cqm5ar7w2p7tuq5twbojyycvol6wugfav3ckq"}}
{
 "thought": "I now have the data to answer",
 "final_answer": "*Instances in compartment Adrian_Birzu (ocid1.compartment.oc1..aaaaaaaagy3yddkkampnhj3cqm5ar7w2p7tuq5twbojyycvol6wugfav3ckq)*\n\nNo compute instances were found in this compartment."
}'''

print("=" * 60)
print("INPUT:")
print("=" * 60)
print(test)
print()

cleaned, table = handler._extract_content_from_response(test)

print("=" * 60)
print("CLEANED OUTPUT:")
print("=" * 60)
print(cleaned)
print()
print(f"Has raw JSON? {'YES ❌' if '{' in cleaned else 'NO ✅'}")
