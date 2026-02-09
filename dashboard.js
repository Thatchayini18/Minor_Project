(function() {
    const run = () => {
        // Get the "permissions" parameter from URL
        const params = new URLSearchParams(window.location.search);
        const permissions = params.get('permissions');

        if (!permissions) return; // Nothing to do if no permissions

        const navButtons = document.querySelectorAll('.nav-btn');

        // Define allowed sections per permission
        const permissionMap = {
            'full': ['overview','traffic','security','environment','services','iot','users','analytics'],
            'monitor,alert,respond': ['overview','traffic','security','environment','services','iot'],
            'monitor,analyze': ['overview','traffic','analytics'],
            'monitor,report': ['overview','environment']
        };

        // Find which key in permissionMap matches the user's permissions
        // Trim whitespace just in case
        const userPermKey = Object.keys(permissionMap).find(key => key === permissions.trim());

        // If no match, hide all (default safe)
        const allowedSections = userPermKey ? permissionMap[userPermKey] : [];

        navButtons.forEach(btn => {
            const sectionId = btn.getAttribute('onclick').match(/'(.+)'/)[1];
            if (allowedSections.includes(sectionId)) {
                btn.style.display = 'flex'; // show
            } else {
                btn.style.display = 'none'; // hide
            }
        });
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', run);
    } else {
        run();
    }
})();
