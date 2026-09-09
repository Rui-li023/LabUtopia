"""Pure contact serialization tests, without an Isaac simulator import."""

import unittest
from types import SimpleNamespace as NS

from experiments.pipette.observe_pipette_replay_contacts import contact_records


def contact(impulse):
    return NS(
        impulse=NS(x=impulse, y=0, z=0),
        position=NS(x=1, y=2, z=3),
        normal=NS(x=0, y=0, z=1),
        separation=-0.0001,
    )


class ContactObserverTests(unittest.TestCase):
    def records(self, headers, contacts):
        return list(contact_records(headers, contacts, str, step=3003, roots=("/World/pipette_01",), threshold=0.001))

    def test_uses_header_slice_and_preserves_action_clock(self):
        header = NS(
            collider0="/World/pipette_01/Body", collider1="/World/rack/Guide", contact_data_offset=1, num_contact_data=1
        )
        records = self.records([header], [contact(50), contact(0.2), contact(99)])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["impulse_Ns"], [0.2, 0, 0])
        self.assertEqual(records[0]["replay_action_step"], 3003)
        self.assertEqual(records[0]["paths"], [header.collider0, header.collider1])

    def test_exact_root_matching_and_strict_threshold(self):
        header = NS(
            collider0="/World/pipette_010/Body",
            collider1="/World/rack/Guide",
            contact_data_offset=0,
            num_contact_data=1,
        )
        self.assertEqual(self.records([header], [contact(0.2)]), [])
        header.collider0 = "/World/pipette_01/Body"
        self.assertEqual(self.records([header], [contact(0.001)]), [])

    def test_rejects_invalid_slice_and_nonfinite_impulse(self):
        header = NS(
            collider0="/World/pipette_01/Body", collider1="/World/rack/Guide", contact_data_offset=1, num_contact_data=1
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            self.records([header], [contact(0.2)])
        header.contact_data_offset = 0
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.records([header], [contact(float("nan"))])
