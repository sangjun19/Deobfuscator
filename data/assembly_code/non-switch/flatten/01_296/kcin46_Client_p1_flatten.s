	.file	"kcin46_Client_p1_flatten.c"
	.text
	.globl	_TIG_IZ_Nb6X_argv
	.bss
	.align 8
	.type	_TIG_IZ_Nb6X_argv, @object
	.size	_TIG_IZ_Nb6X_argv, 8
_TIG_IZ_Nb6X_argv:
	.zero	8
	.globl	_TIG_IZ_Nb6X_argc
	.align 4
	.type	_TIG_IZ_Nb6X_argc, @object
	.size	_TIG_IZ_Nb6X_argc, 4
_TIG_IZ_Nb6X_argc:
	.zero	4
	.globl	_TIG_IZ_Nb6X_envp
	.align 8
	.type	_TIG_IZ_Nb6X_envp, @object
	.size	_TIG_IZ_Nb6X_envp, 8
_TIG_IZ_Nb6X_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Malloc failed for inputBuff\n"
.LC1:
	.string	"read fail"
	.align 8
.LC2:
	.string	"The getaddrinfo() function failed\n"
	.align 8
.LC3:
	.string	"Shutdown for writing to the socket failed\n"
	.align 8
.LC4:
	.string	"realloc for final inputBuff didn't work\n"
.LC5:
	.string	"%s"
	.align 8
.LC6:
	.string	"Reallocation of memory failed while reading from stdin\n"
.LC7:
	.string	"Connect failed\n"
.LC8:
	.string	"Writing to the socket failed!"
	.align 8
.LC9:
	.string	"Realloc failed inside while loop for reading from the socket"
	.align 8
.LC10:
	.string	"Malloc failed for the buffer for reading from the socket\n"
.LC11:
	.string	"inet_pton failed\n"
	.align 8
.LC12:
	.string	"Incorrect number of arguments!\n"
.LC13:
	.string	"Invalid address string\n"
.LC14:
	.string	"Creating a socket failed\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$320, %rsp
	movl	%edi, -292(%rbp)
	movq	%rsi, -304(%rbp)
	movq	%rdx, -312(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_Nb6X_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Nb6X_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Nb6X_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 131 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Nb6X--0
# 0 "" 2
#NO_APP
	movl	-292(%rbp), %eax
	movl	%eax, _TIG_IZ_Nb6X_argc(%rip)
	movq	-304(%rbp), %rax
	movq	%rax, _TIG_IZ_Nb6X_argv(%rip)
	movq	-312(%rbp), %rax
	movq	%rax, _TIG_IZ_Nb6X_envp(%rip)
	nop
	movq	$45, -152(%rbp)
.L92:
	cmpq	$78, -152(%rbp)
	ja	.L94
	movq	-152(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L56-.L8
	.long	.L55-.L8
	.long	.L94-.L8
	.long	.L54-.L8
	.long	.L53-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L52-.L8
	.long	.L94-.L8
	.long	.L51-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L50-.L8
	.long	.L49-.L8
	.long	.L94-.L8
	.long	.L48-.L8
	.long	.L47-.L8
	.long	.L46-.L8
	.long	.L94-.L8
	.long	.L45-.L8
	.long	.L94-.L8
	.long	.L44-.L8
	.long	.L43-.L8
	.long	.L42-.L8
	.long	.L94-.L8
	.long	.L41-.L8
	.long	.L40-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L39-.L8
	.long	.L38-.L8
	.long	.L37-.L8
	.long	.L36-.L8
	.long	.L94-.L8
	.long	.L35-.L8
	.long	.L34-.L8
	.long	.L94-.L8
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L94-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L94-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L94-.L8
	.long	.L26-.L8
	.long	.L94-.L8
	.long	.L25-.L8
	.long	.L94-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L94-.L8
	.long	.L21-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L94-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L94-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L41:
	cmpl	$0, -264(%rbp)
	jns	.L57
	movq	$60, -152(%rbp)
	jmp	.L59
.L57:
	movq	$9, -152(%rbp)
	jmp	.L59
.L25:
	cmpq	$0, -224(%rbp)
	jne	.L60
	movq	$30, -152(%rbp)
	jmp	.L59
.L60:
	movq	$37, -152(%rbp)
	jmp	.L59
.L23:
	movq	$0, -216(%rbp)
	movq	$255, -208(%rbp)
	movq	$19, -152(%rbp)
	jmp	.L59
.L53:
	movq	-168(%rbp), %rcx
	movl	-272(%rbp), %eax
	movl	$8, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -96(%rbp)
	movq	-96(%rbp), %rax
	movl	%eax, -252(%rbp)
	movq	$34, -152(%rbp)
	jmp	.L59
.L38:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$28, %edx
	movl	$1, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L48:
	leaq	-80(%rbp), %rax
	movl	$48, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movl	$0, -76(%rbp)
	movl	$1, -72(%rbp)
	movl	$6, -68(%rbp)
	leaq	-232(%rbp), %rcx
	leaq	-80(%rbp), %rdx
	movq	-192(%rbp), %rsi
	movq	-200(%rbp), %rax
	movq	%rax, %rdi
	call	getaddrinfo@PLT
	movl	%eax, -240(%rbp)
	movl	-240(%rbp), %eax
	movl	%eax, -268(%rbp)
	movq	$13, -152(%rbp)
	jmp	.L59
.L37:
	movl	-272(%rbp), %eax
	movl	$1, %esi
	movl	%eax, %edi
	call	shutdown@PLT
	movl	%eax, -256(%rbp)
	movq	$69, -152(%rbp)
	jmp	.L59
.L50:
	leaq	-32(%rbp), %rax
	movl	$16, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movw	$2, -32(%rbp)
	movq	-232(%rbp), %rax
	movq	24(%rax), %rax
	movq	%rax, -112(%rbp)
	movq	-112(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, %edi
	call	inet_ntoa@PLT
	movq	%rax, -104(%rbp)
	leaq	-32(%rbp), %rax
	leaq	4(%rax), %rdx
	movq	-104(%rbp), %rax
	movq	%rax, %rsi
	movl	$2, %edi
	call	inet_pton@PLT
	movl	%eax, -236(%rbp)
	movl	-236(%rbp), %eax
	movl	%eax, -264(%rbp)
	movq	$42, -152(%rbp)
	jmp	.L59
.L12:
	cmpl	$0, -256(%rbp)
	jns	.L62
	movq	$3, -152(%rbp)
	jmp	.L59
.L62:
	movq	$67, -152(%rbp)
	jmp	.L59
.L27:
	movq	$52, -152(%rbp)
	jmp	.L59
.L7:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$9, %edx
	movl	$1, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L55:
	cmpl	$0, -272(%rbp)
	jns	.L64
	movq	$29, -152(%rbp)
	jmp	.L59
.L64:
	movq	$15, -152(%rbp)
	jmp	.L59
.L42:
	movq	-216(%rbp), %rax
	movq	%rax, -120(%rbp)
	addq	$1, -216(%rbp)
	movq	-224(%rbp), %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movl	-276(%rbp), %edx
	movb	%dl, (%rax)
	movq	$72, -152(%rbp)
	jmp	.L59
.L9:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$34, %edx
	movl	$1, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L54:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$42, %edx
	movl	$1, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L47:
	cmpq	$0, -224(%rbp)
	jne	.L66
	movq	$26, -152(%rbp)
	jmp	.L59
.L66:
	movq	$37, -152(%rbp)
	jmp	.L59
.L44:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$40, %edx
	movl	$1, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L13:
	movq	-168(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-168(%rbp), %rax
	movl	$8, %esi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -160(%rbp)
	movq	$7, -152(%rbp)
	jmp	.L59
.L40:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$55, %edx
	movl	$1, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L51:
	movzwl	-278(%rbp), %eax
	movl	%eax, %edi
	call	htons@PLT
	movw	%ax, -30(%rbp)
	leaq	-32(%rbp), %rcx
	movl	-272(%rbp), %eax
	movl	$16, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	connect@PLT
	movl	%eax, -260(%rbp)
	movq	$64, -152(%rbp)
	jmp	.L59
.L49:
	cmpl	$0, -268(%rbp)
	je	.L68
	movq	$77, -152(%rbp)
	jmp	.L59
.L68:
	movq	$12, -152(%rbp)
	jmp	.L59
.L17:
	cmpq	$0, -184(%rbp)
	jne	.L70
	movq	$21, -152(%rbp)
	jmp	.L59
.L70:
	movq	$51, -152(%rbp)
	jmp	.L59
.L24:
	movq	-216(%rbp), %rdx
	movq	-224(%rbp), %rcx
	movl	-272(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	%rax, -176(%rbp)
	movq	$47, -152(%rbp)
	jmp	.L59
.L45:
	cmpl	$3, -292(%rbp)
	je	.L72
	movq	$22, -152(%rbp)
	jmp	.L59
.L72:
	movq	$41, -152(%rbp)
	jmp	.L59
.L36:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$15, %edx
	movl	$1, %esi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L46:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	exit@PLT
.L31:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$60, %edx
	movl	$1, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L14:
	movl	$1024, %edi
	call	malloc@PLT
	movq	%rax, -144(%rbp)
	movq	-144(%rbp), %rax
	movq	%rax, -168(%rbp)
	movq	$73, -152(%rbp)
	jmp	.L59
.L21:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$57, %edx
	movl	$1, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L19:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$17, %edx
	movl	$1, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L20:
	movq	-208(%rbp), %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movq	%rax, -208(%rbp)
	movq	-208(%rbp), %rdx
	movq	-224(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -128(%rbp)
	movq	-128(%rbp), %rax
	movq	%rax, -224(%rbp)
	movq	$16, -152(%rbp)
	jmp	.L59
.L32:
	cmpl	$-1, -276(%rbp)
	je	.L74
	movq	$23, -152(%rbp)
	jmp	.L59
.L74:
	movq	$0, -152(%rbp)
	jmp	.L59
.L18:
	movq	-208(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -136(%rbp)
	movq	-136(%rbp), %rax
	movq	%rax, -224(%rbp)
	movq	$49, -152(%rbp)
	jmp	.L59
.L35:
	cmpl	$0, -252(%rbp)
	je	.L76
	movq	$66, -152(%rbp)
	jmp	.L59
.L76:
	movq	$35, -152(%rbp)
	jmp	.L59
.L43:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$31, %edx
	movl	$1, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L22:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$23, %edx
	movl	$1, %esi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L26:
	cmpq	$0, -176(%rbp)
	jns	.L78
	movq	$17, -152(%rbp)
	jmp	.L59
.L78:
	movq	$31, -152(%rbp)
	jmp	.L59
.L10:
	cmpq	$0, -168(%rbp)
	jne	.L80
	movq	$55, -152(%rbp)
	jmp	.L59
.L80:
	movq	$4, -152(%rbp)
	jmp	.L59
.L28:
	movq	-168(%rbp), %rax
	movl	$8, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movq	$4, -152(%rbp)
	jmp	.L59
.L11:
	movq	-216(%rbp), %rax
	cmpq	-208(%rbp), %rax
	jne	.L82
	movq	$59, -152(%rbp)
	jmp	.L59
.L82:
	movq	$37, -152(%rbp)
	jmp	.L59
.L33:
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -276(%rbp)
	movq	$38, -152(%rbp)
	jmp	.L59
.L16:
	cmpl	$0, -260(%rbp)
	jns	.L84
	movq	$32, -152(%rbp)
	jmp	.L59
.L84:
	movq	$61, -152(%rbp)
	jmp	.L59
.L30:
	movq	-304(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -200(%rbp)
	movq	-304(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -192(%rbp)
	movq	-304(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -248(%rbp)
	movl	-248(%rbp), %eax
	movw	%ax, -278(%rbp)
	movl	$6, %edx
	movl	$1, %esi
	movl	$2, %edi
	call	socket@PLT
	movl	%eax, -244(%rbp)
	movl	-244(%rbp), %eax
	movl	%eax, -272(%rbp)
	movq	$1, -152(%rbp)
	jmp	.L59
.L29:
	cmpl	$0, -264(%rbp)
	jne	.L86
	movq	$53, -152(%rbp)
	jmp	.L59
.L86:
	movq	$25, -152(%rbp)
	jmp	.L59
.L56:
	movq	-216(%rbp), %rax
	movq	%rax, -88(%rbp)
	addq	$1, -216(%rbp)
	movq	-224(%rbp), %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	-216(%rbp), %rdx
	movq	-224(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -184(%rbp)
	movq	$63, -152(%rbp)
	jmp	.L59
.L15:
	cmpl	$0, -252(%rbp)
	jns	.L88
	movq	$78, -152(%rbp)
	jmp	.L59
.L88:
	movq	$68, -152(%rbp)
	jmp	.L59
.L52:
	cmpq	$0, -160(%rbp)
	jne	.L90
	movq	$40, -152(%rbp)
	jmp	.L59
.L90:
	movq	$44, -152(%rbp)
	jmp	.L59
.L34:
	movl	-272(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	$0, %edi
	call	exit@PLT
.L39:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$25, %edx
	movl	$1, %esi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L94:
	nop
.L59:
	jmp	.L92
	.cfi_endproc
.LFE0:
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
