	.file	"nim3xh_chat---Network-Chat-Client-C-Programming_client_flatten.c"
	.text
	.globl	_TIG_IZ_EsYa_argc
	.bss
	.align 4
	.type	_TIG_IZ_EsYa_argc, @object
	.size	_TIG_IZ_EsYa_argc, 4
_TIG_IZ_EsYa_argc:
	.zero	4
	.globl	_TIG_IZ_EsYa_argv
	.align 8
	.type	_TIG_IZ_EsYa_argv, @object
	.size	_TIG_IZ_EsYa_argv, 8
_TIG_IZ_EsYa_argv:
	.zero	8
	.globl	_TIG_IZ_EsYa_envp
	.align 8
	.type	_TIG_IZ_EsYa_envp, @object
	.size	_TIG_IZ_EsYa_envp, 8
_TIG_IZ_EsYa_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"quit %s"
	.text
	.globl	sendQuitMessage
	.type	sendQuitMessage, @function
sendQuitMessage:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1072, %rsp
	movl	%edi, -1060(%rbp)
	movq	%rsi, -1072(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -1056(%rbp)
.L7:
	cmpq	$2, -1056(%rbp)
	je	.L2
	cmpq	$2, -1056(%rbp)
	ja	.L10
	cmpq	$0, -1056(%rbp)
	je	.L11
	cmpq	$1, -1056(%rbp)
	jne	.L10
	movq	$2, -1056(%rbp)
	jmp	.L5
.L2:
	movq	-1072(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rdx, %rcx
	leaq	.LC0(%rip), %rdx
	movl	$1024, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1048(%rbp)
	movq	-1048(%rbp), %rdx
	leaq	-1040(%rbp), %rsi
	movl	-1060(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$0, -1056(%rbp)
	jmp	.L5
.L10:
	nop
.L5:
	jmp	.L7
.L11:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L9
	call	__stack_chk_fail@PLT
.L9:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	sendQuitMessage, .-sendQuitMessage
	.section	.rodata
	.align 8
.LC1:
	.string	"%s has changed their name to %s"
	.text
	.globl	sendNameChange
	.type	sendNameChange, @function
sendNameChange:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1088, %rsp
	movl	%edi, -1060(%rbp)
	movq	%rsi, -1072(%rbp)
	movq	%rdx, -1080(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -1056(%rbp)
.L18:
	cmpq	$2, -1056(%rbp)
	je	.L21
	cmpq	$2, -1056(%rbp)
	ja	.L22
	cmpq	$0, -1056(%rbp)
	je	.L15
	cmpq	$1, -1056(%rbp)
	jne	.L22
	movq	$0, -1056(%rbp)
	jmp	.L16
.L15:
	movq	-1080(%rbp), %rcx
	movq	-1072(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC1(%rip), %rdx
	movl	$1024, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1048(%rbp)
	movq	-1048(%rbp), %rdx
	leaq	-1040(%rbp), %rsi
	movl	-1060(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$2, -1056(%rbp)
	jmp	.L16
.L22:
	nop
.L16:
	jmp	.L18
.L21:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L20
	call	__stack_chk_fail@PLT
.L20:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	sendNameChange, .-sendNameChange
	.globl	setNonBlocking
	.type	setNonBlocking, @function
setNonBlocking:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L29:
	cmpq	$2, -8(%rbp)
	je	.L24
	cmpq	$2, -8(%rbp)
	ja	.L30
	cmpq	$0, -8(%rbp)
	je	.L31
	cmpq	$1, -8(%rbp)
	jne	.L30
	movq	$2, -8(%rbp)
	jmp	.L27
.L24:
	movl	-20(%rbp), %eax
	movl	$0, %edx
	movl	$3, %esi
	movl	%eax, %edi
	movl	$0, %eax
	call	fcntl@PLT
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	orb	$8, %ah
	movl	%eax, %edx
	movl	-20(%rbp), %eax
	movl	$4, %esi
	movl	%eax, %edi
	movl	$0, %eax
	call	fcntl@PLT
	movq	$0, -8(%rbp)
	jmp	.L27
.L30:
	nop
.L27:
	jmp	.L29
.L31:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	setNonBlocking, .-setNonBlocking
	.section	.rodata
.LC2:
	.string	"%s"
	.align 8
.LC3:
	.string	"%s has changed their name to %s\n"
	.align 8
.LC4:
	.string	"Usage: %s <ip_address> <port>\n"
	.align 8
.LC5:
	.string	"Invalid name format. Usage: name <new_name>"
.LC6:
	.string	"Socket creation failed"
.LC7:
	.string	"\n"
.LC8:
	.string	"quit"
.LC9:
	.string	"Connection failed"
	.align 8
.LC10:
	.string	"Server disconnected unexpectedly. Exiting..."
.LC11:
	.string	"Receive error"
.LC12:
	.string	"Sent quit message. Exiting..."
.LC13:
	.string	"name"
.LC14:
	.string	"Connected to the server"
.LC15:
	.string	" "
.LC16:
	.string	"%s: %s"
	.text
	.globl	main
	.type	main, @function
main:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$2400, %rsp
	movl	%edi, -2372(%rbp)
	movq	%rsi, -2384(%rbp)
	movq	%rdx, -2392(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_EsYa_envp(%rip)
	nop
.L33:
	movq	$0, _TIG_IZ_EsYa_argv(%rip)
	nop
.L34:
	movl	$0, _TIG_IZ_EsYa_argc(%rip)
	nop
	nop
.L35:
.L36:
#APP
# 106 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-EsYa--0
# 0 "" 2
#NO_APP
	movl	-2372(%rbp), %eax
	movl	%eax, _TIG_IZ_EsYa_argc(%rip)
	movq	-2384(%rbp), %rax
	movq	%rax, _TIG_IZ_EsYa_argv(%rip)
	movq	-2392(%rbp), %rax
	movq	%rax, _TIG_IZ_EsYa_envp(%rip)
	nop
	movq	$54, -2312(%rbp)
.L97:
	cmpq	$54, -2312(%rbp)
	ja	.L100
	movq	-2312(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L39(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L39(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L39:
	.long	.L100-.L39
	.long	.L100-.L39
	.long	.L72-.L39
	.long	.L71-.L39
	.long	.L70-.L39
	.long	.L69-.L39
	.long	.L100-.L39
	.long	.L68-.L39
	.long	.L67-.L39
	.long	.L100-.L39
	.long	.L66-.L39
	.long	.L100-.L39
	.long	.L65-.L39
	.long	.L64-.L39
	.long	.L63-.L39
	.long	.L100-.L39
	.long	.L100-.L39
	.long	.L62-.L39
	.long	.L61-.L39
	.long	.L60-.L39
	.long	.L100-.L39
	.long	.L59-.L39
	.long	.L58-.L39
	.long	.L57-.L39
	.long	.L100-.L39
	.long	.L56-.L39
	.long	.L100-.L39
	.long	.L55-.L39
	.long	.L54-.L39
	.long	.L53-.L39
	.long	.L52-.L39
	.long	.L100-.L39
	.long	.L100-.L39
	.long	.L51-.L39
	.long	.L100-.L39
	.long	.L100-.L39
	.long	.L50-.L39
	.long	.L100-.L39
	.long	.L100-.L39
	.long	.L49-.L39
	.long	.L100-.L39
	.long	.L48-.L39
	.long	.L47-.L39
	.long	.L100-.L39
	.long	.L46-.L39
	.long	.L100-.L39
	.long	.L45-.L39
	.long	.L44-.L39
	.long	.L100-.L39
	.long	.L43-.L39
	.long	.L42-.L39
	.long	.L41-.L39
	.long	.L100-.L39
	.long	.L40-.L39
	.long	.L38-.L39
	.text
.L61:
	leaq	-2128(%rbp), %rax
	movl	$1024, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	leaq	-2128(%rbp), %rsi
	movl	-2368(%rbp), %eax
	movl	$0, %ecx
	movl	$1024, %edx
	movl	%eax, %edi
	call	recv@PLT
	movq	%rax, -2288(%rbp)
	movq	-2288(%rbp), %rax
	movq	%rax, -2344(%rbp)
	movq	$49, -2312(%rbp)
	jmp	.L73
.L42:
	leaq	-2256(%rbp), %rcx
	leaq	-2192(%rbp), %rax
	movl	$64, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncpy@PLT
	movq	-2328(%rbp), %rdx
	leaq	-2256(%rbp), %rax
	movq	%rdx, %rcx
	leaq	.LC2(%rip), %rdx
	movl	$64, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-2256(%rbp), %rdx
	leaq	-2192(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-2256(%rbp), %rdx
	leaq	-2192(%rbp), %rcx
	movl	-2368(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	sendNameChange
	movq	$18, -2312(%rbp)
	jmp	.L73
.L56:
	movq	-2384(%rbp), %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC4(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$1, %edi
	call	exit@PLT
.L43:
	cmpq	$0, -2344(%rbp)
	jle	.L74
	movq	$2, -2312(%rbp)
	jmp	.L73
.L74:
	movq	$13, -2312(%rbp)
	jmp	.L73
.L70:
	cmpl	$-1, -2368(%rbp)
	jne	.L76
	movq	$36, -2312(%rbp)
	jmp	.L73
.L76:
	movq	$5, -2312(%rbp)
	jmp	.L73
.L52:
	movl	-2368(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$51, -2312(%rbp)
	jmp	.L73
.L63:
	call	__errno_location@PLT
	movq	%rax, -2336(%rbp)
	movq	$12, -2312(%rbp)
	jmp	.L73
.L65:
	movq	-2336(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$11, %eax
	je	.L78
	movq	$53, -2312(%rbp)
	jmp	.L73
.L78:
	movq	$33, -2312(%rbp)
	jmp	.L73
.L67:
	cmpl	$63, -2364(%rbp)
	jbe	.L80
	movq	$47, -2312(%rbp)
	jmp	.L73
.L80:
	movq	$41, -2312(%rbp)
	jmp	.L73
.L38:
	cmpl	$3, -2372(%rbp)
	je	.L82
	movq	$25, -2312(%rbp)
	jmp	.L73
.L82:
	movq	$42, -2312(%rbp)
	jmp	.L73
.L57:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$18, -2312(%rbp)
	jmp	.L73
.L71:
	cmpl	$0, -2356(%rbp)
	jne	.L84
	movq	$7, -2312(%rbp)
	jmp	.L73
.L84:
	movq	$29, -2312(%rbp)
	jmp	.L73
.L59:
	cmpl	$0, -2352(%rbp)
	jne	.L86
	movq	$10, -2312(%rbp)
	jmp	.L73
.L86:
	movq	$46, -2312(%rbp)
	jmp	.L73
.L50:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L64:
	cmpq	$0, -2344(%rbp)
	jne	.L88
	movq	$22, -2312(%rbp)
	jmp	.L73
.L88:
	movq	$14, -2312(%rbp)
	jmp	.L73
.L41:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L98
	jmp	.L99
.L60:
	leaq	-2128(%rbp), %rax
	leaq	.LC7(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -2296(%rbp)
	leaq	-2128(%rbp), %rdx
	movq	-2296(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-2128(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -2352(%rbp)
	movq	$21, -2312(%rbp)
	jmp	.L73
.L62:
	cmpl	$-1, -2360(%rbp)
	jne	.L91
	movq	$27, -2312(%rbp)
	jmp	.L73
.L91:
	movq	$39, -2312(%rbp)
	jmp	.L73
.L55:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	-2368(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	$1, %edi
	call	exit@PLT
.L58:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$30, -2312(%rbp)
	jmp	.L73
.L54:
	cmpq	$0, -2320(%rbp)
	jle	.L93
	movq	$19, -2312(%rbp)
	jmp	.L73
.L93:
	movq	$18, -2312(%rbp)
	jmp	.L73
.L40:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$30, -2312(%rbp)
	jmp	.L73
.L44:
	movl	$0, %edx
	movl	$1, %esi
	movl	$2, %edi
	call	socket@PLT
	movl	%eax, -2368(%rbp)
	movq	$4, -2312(%rbp)
	jmp	.L73
.L46:
	cmpq	$0, -2328(%rbp)
	je	.L95
	movq	$50, -2312(%rbp)
	jmp	.L73
.L95:
	movq	$23, -2312(%rbp)
	jmp	.L73
.L69:
	leaq	-2272(%rbp), %rax
	movl	$16, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movw	$2, -2272(%rbp)
	movq	-2384(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	inet_addr@PLT
	movl	%eax, -2268(%rbp)
	movq	-2384(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -2348(%rbp)
	movl	-2348(%rbp), %eax
	movzwl	%ax, %eax
	movl	%eax, %edi
	call	htons@PLT
	movw	%ax, -2270(%rbp)
	leaq	-2272(%rbp), %rcx
	movl	-2368(%rbp), %eax
	movl	$16, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	connect@PLT
	movl	%eax, -2360(%rbp)
	movq	$17, -2312(%rbp)
	jmp	.L73
.L51:
	leaq	-2128(%rbp), %rax
	movl	$1024, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	leaq	-2128(%rbp), %rax
	movl	$1024, %edx
	movq	%rax, %rsi
	movl	$0, %edi
	call	read@PLT
	movq	%rax, -2320(%rbp)
	movq	$28, -2312(%rbp)
	jmp	.L73
.L48:
	movl	-2364(%rbp), %eax
	movb	$0, -2256(%rbp,%rax)
	addl	$1, -2364(%rbp)
	movq	$8, -2312(%rbp)
	jmp	.L73
.L66:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-2256(%rbp), %rdx
	movl	-2368(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	sendQuitMessage
	movq	$30, -2312(%rbp)
	jmp	.L73
.L47:
	movb	$85, -2256(%rbp)
	movb	$115, -2255(%rbp)
	movb	$101, -2254(%rbp)
	movb	$114, -2253(%rbp)
	movb	$0, -2252(%rbp)
	movl	$5, -2364(%rbp)
	movq	$8, -2312(%rbp)
	jmp	.L73
.L45:
	leaq	-2128(%rbp), %rax
	movl	$4, %edx
	leaq	.LC13(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -2356(%rbp)
	movq	$3, -2312(%rbp)
	jmp	.L73
.L49:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %edi
	call	setNonBlocking
	movl	-2368(%rbp), %eax
	movl	%eax, %edi
	call	setNonBlocking
	movq	$18, -2312(%rbp)
	jmp	.L73
.L68:
	leaq	-2128(%rbp), %rax
	leaq	.LC15(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -2280(%rbp)
	movq	-2280(%rbp), %rax
	movq	%rax, -2328(%rbp)
	leaq	.LC15(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -2328(%rbp)
	movq	$44, -2312(%rbp)
	jmp	.L73
.L53:
	leaq	-2128(%rbp), %rcx
	leaq	-2256(%rbp), %rdx
	leaq	-1104(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC16(%rip), %rdx
	movl	$1091, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1104(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2304(%rbp)
	movq	-2304(%rbp), %rdx
	leaq	-1104(%rbp), %rsi
	movl	-2368(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$18, -2312(%rbp)
	jmp	.L73
.L72:
	leaq	-2128(%rbp), %rdx
	movq	-2344(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-2128(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$33, -2312(%rbp)
	jmp	.L73
.L100:
	nop
.L73:
	jmp	.L97
.L99:
	call	__stack_chk_fail@PLT
.L98:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
