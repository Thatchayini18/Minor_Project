document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const errorMessage = document.getElementById('errorMessage');
    const passwordInput = document.getElementById('password');
    const passwordToggle = document.querySelector('.password-toggle');
    const eyeIcon = document.getElementById('eyeIcon');

    // Password visibility toggle
    if (passwordToggle) {
        passwordToggle.addEventListener('click', function() {
            const isPassword = passwordInput.type === 'password';
            passwordInput.type = isPassword ? 'text' : 'password';
            
            if (isPassword) {
                eyeIcon.innerHTML = `
                    <path stroke-linecap="round" stroke-linejoin="round" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                `;
            } else {
                eyeIcon.innerHTML = `
                    <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path stroke-linecap="round" stroke-linejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                `;
            }
        });
    }

    // Form submission handler
    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const email = document.getElementById('email').value.trim();
        const password = passwordInput.value;

        // Hide any previous error
        hideError();

        try {
            // Fetch and parse the XML file
            const response = await fetch('./smartcity.xml');
            
            if (!response.ok) {
                throw new Error('Could not load user database');
            }

            const xmlText = await response.text();
            const parser = new DOMParser();
            const xmlDoc = parser.parseFromString(xmlText, 'application/xml');

            // Check for XML parsing errors
            const parseError = xmlDoc.querySelector('parsererror');
            if (parseError) {
                throw new Error('Error parsing user database');
            }

            // Find all user elements
            const users = xmlDoc.querySelectorAll('smartCity > users > user');
            let authenticated = false;
            let userPermissions = '';

            // Check each user
            users.forEach(function(user) {
                const userPassword = user.getAttribute('password');
                const userEmail = user.querySelector('email');
                const permissionsNode = user.querySelector('permissions');

                if (
                    userEmail &&
                    userEmail.textContent.trim() === email &&
                    userPassword === password
                ) {
                    authenticated = true;
                    userPermissions = permissionsNode
                        ? permissionsNode.textContent.trim()
                        : '';
                }
            });

            if (authenticated) {
                // Store login state (optional)
                sessionStorage.setItem('Ecocity_authenticated', 'true');
                sessionStorage.setItem('Ecocity_user', email);

                // Redirect with permissions in URL
                const params = new URLSearchParams({
                    permissions: userPermissions
                });
                
                window.location.href = `smartcity.xml?${params.toString()}`;
            } else {
                showError('Invalid email or password. Please try again.');
            }

        } catch (error) {
            console.error('Login error:', error);
            showError('An error occurred. Please try again later.');
        }
    });

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }

    function hideError() {
        errorMessage.style.display = 'none';
    }
});
