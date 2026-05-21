# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================

# Intentionally empty: the URL adapter is discovered via the ``tlc.url_adapters`` entry point
# declared in ``pyproject.toml``. Eagerly importing ``adapter`` here would cause a circular
# import when ``python -m digifarm_virtual_table.create_table`` loads the package before
# ``tlc`` itself has been imported.
