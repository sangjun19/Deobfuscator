	.file	"rivoreo_ToolBox_r_flatten.c"
	.text
	.globl	_TIG_IZ_bBRm_argv
	.bss
	.align 8
	.type	_TIG_IZ_bBRm_argv, @object
	.size	_TIG_IZ_bBRm_argv, 8
_TIG_IZ_bBRm_argv:
	.zero	8
	.globl	_TIG_IZ_bBRm_envp
	.align 8
	.type	_TIG_IZ_bBRm_envp, @object
	.size	_TIG_IZ_bBRm_envp, 8
_TIG_IZ_bBRm_envp:
	.zero	8
	.globl	_TIG_IZ_bBRm_argc
	.align 4
	.type	_TIG_IZ_bBRm_argc, @object
	.size	_TIG_IZ_bBRm_argc, 4
_TIG_IZ_bBRm_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"cannot open /dev/mem\n"
.LC1:
	.string	"-b"
.LC2:
	.string	"%08x: %02x\n"
.LC3:
	.string	"cannot mmap region\n"
.LC4:
	.string	"%08x: %08x\n"
.LC5:
	.string	"-h"
.LC6:
	.string	"-s"
.LC7:
	.string	"%08x: %04x\n"
.LC8:
	.string	"invalid end address\n"
.LC9:
	.string	"/dev/mem"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movl	%edi, -132(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%rdx, -152(%rbp)
	movq	$0, _TIG_IZ_bBRm_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_bBRm_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_bBRm_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 103 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-bBRm--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_bBRm_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_bBRm_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_bBRm_envp(%rip)
	nop
	movq	$39, -48(%rbp)
.L101:
	cmpq	$64, -48(%rbp)
	ja	.L102
	movq	-48(%rbp), %rax
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
	.long	.L102-.L8
	.long	.L102-.L8
	.long	.L61-.L8
	.long	.L60-.L8
	.long	.L59-.L8
	.long	.L58-.L8
	.long	.L57-.L8
	.long	.L56-.L8
	.long	.L102-.L8
	.long	.L55-.L8
	.long	.L54-.L8
	.long	.L53-.L8
	.long	.L52-.L8
	.long	.L102-.L8
	.long	.L51-.L8
	.long	.L50-.L8
	.long	.L102-.L8
	.long	.L102-.L8
	.long	.L49-.L8
	.long	.L48-.L8
	.long	.L47-.L8
	.long	.L46-.L8
	.long	.L102-.L8
	.long	.L45-.L8
	.long	.L44-.L8
	.long	.L43-.L8
	.long	.L42-.L8
	.long	.L41-.L8
	.long	.L40-.L8
	.long	.L39-.L8
	.long	.L38-.L8
	.long	.L37-.L8
	.long	.L102-.L8
	.long	.L36-.L8
	.long	.L35-.L8
	.long	.L34-.L8
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L102-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L102-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L102-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L49:
	movl	-112(%rbp), %eax
	cmpl	-104(%rbp), %eax
	ja	.L62
	movq	$38, -48(%rbp)
	jmp	.L64
.L62:
	movq	$28, -48(%rbp)
	jmp	.L64
.L21:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$21, %edx
	movl	$1, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$19, -48(%rbp)
	jmp	.L64
.L43:
	movq	-144(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC1(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -92(%rbp)
	movq	$6, -48(%rbp)
	jmp	.L64
.L22:
	cmpl	$2, -132(%rbp)
	jle	.L65
	movq	$35, -48(%rbp)
	jmp	.L64
.L65:
	movq	$20, -48(%rbp)
	jmp	.L64
.L19:
	movq	-72(%rbp), %rax
	movl	-108(%rbp), %edx
	movl	%edx, (%rax)
	movq	$24, -48(%rbp)
	jmp	.L64
.L59:
	cmpl	$0, -120(%rbp)
	je	.L67
	movq	$52, -48(%rbp)
	jmp	.L64
.L67:
	movq	$24, -48(%rbp)
	jmp	.L64
.L38:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %ecx
	movq	stderr(%rip), %rax
	movl	-112(%rbp), %edx
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$53, -48(%rbp)
	jmp	.L64
.L10:
	movq	$53, -48(%rbp)
	jmp	.L64
.L51:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$19, %edx
	movl	$1, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$29, -48(%rbp)
	jmp	.L64
.L50:
	cmpq	$0, -80(%rbp)
	je	.L69
	movq	$61, -48(%rbp)
	jmp	.L64
.L69:
	movq	$12, -48(%rbp)
	jmp	.L64
.L15:
	call	usage
	movq	$7, -48(%rbp)
	jmp	.L64
.L37:
	cmpl	$2, -132(%rbp)
	jg	.L71
	movq	$34, -48(%rbp)
	jmp	.L64
.L71:
	movq	$25, -48(%rbp)
	jmp	.L64
.L52:
	cmpl	$0, -104(%rbp)
	jne	.L73
	movq	$41, -48(%rbp)
	jmp	.L64
.L73:
	movq	$59, -48(%rbp)
	jmp	.L64
.L25:
	movq	-144(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	jne	.L75
	movq	$31, -48(%rbp)
	jmp	.L64
.L75:
	movq	$25, -48(%rbp)
	jmp	.L64
.L17:
	movl	$0, %eax
	jmp	.L77
.L45:
	cmpl	$-1, -116(%rbp)
	jne	.L78
	movq	$50, -48(%rbp)
	jmp	.L64
.L78:
	movq	$2, -48(%rbp)
	jmp	.L64
.L60:
	cmpq	$-1, -88(%rbp)
	jne	.L80
	movq	$14, -48(%rbp)
	jmp	.L64
.L80:
	movq	$18, -48(%rbp)
	jmp	.L64
.L44:
	movq	-72(%rbp), %rax
	movl	(%rax), %ecx
	movq	stderr(%rip), %rax
	movl	-112(%rbp), %edx
	leaq	.LC4(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$53, -48(%rbp)
	jmp	.L64
.L46:
	movq	-144(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC5(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -100(%rbp)
	movq	$51, -48(%rbp)
	jmp	.L64
.L33:
	movl	-108(%rbp), %eax
	movl	%eax, %edx
	movq	-56(%rbp), %rax
	movb	%dl, (%rax)
	movq	$30, -48(%rbp)
	jmp	.L64
.L14:
	cmpl	$0, -96(%rbp)
	jne	.L82
	movq	$43, -48(%rbp)
	jmp	.L64
.L82:
	movq	$21, -48(%rbp)
	jmp	.L64
.L42:
	cmpl	$0, -120(%rbp)
	je	.L84
	movq	$36, -48(%rbp)
	jmp	.L64
.L84:
	movq	$30, -48(%rbp)
	jmp	.L64
.L53:
	movl	$-1, %eax
	jmp	.L77
.L55:
	movq	-144(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -96(%rbp)
	movq	$57, -48(%rbp)
	jmp	.L64
.L9:
	movl	$4, -124(%rbp)
	movl	$0, -120(%rbp)
	movl	$0, -104(%rbp)
	movq	$33, -48(%rbp)
	jmp	.L64
.L20:
	cmpl	$0, -100(%rbp)
	jne	.L86
	movq	$46, -48(%rbp)
	jmp	.L64
.L86:
	movq	$5, -48(%rbp)
	jmp	.L64
.L48:
	movl	$-1, %eax
	jmp	.L77
.L16:
	movl	$-1, %eax
	jmp	.L77
.L12:
	movl	-112(%rbp), %eax
	andl	$4095, %eax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -72(%rbp)
	movq	$4, -48(%rbp)
	jmp	.L64
.L13:
	movl	-104(%rbp), %eax
	cmpl	-112(%rbp), %eax
	ja	.L88
	movq	$10, -48(%rbp)
	jmp	.L64
.L88:
	movq	$49, -48(%rbp)
	jmp	.L64
.L57:
	cmpl	$0, -92(%rbp)
	jne	.L90
	movq	$44, -48(%rbp)
	jmp	.L64
.L90:
	movq	$9, -48(%rbp)
	jmp	.L64
.L41:
	movl	-108(%rbp), %eax
	movl	%eax, %edx
	movq	-64(%rbp), %rax
	movw	%dx, (%rax)
	movq	$47, -48(%rbp)
	jmp	.L64
.L31:
	cmpl	$4, -124(%rbp)
	je	.L92
	cmpl	$4, -124(%rbp)
	jg	.L93
	cmpl	$1, -124(%rbp)
	je	.L94
	cmpl	$2, -124(%rbp)
	je	.L95
	jmp	.L93
.L94:
	movq	$42, -48(%rbp)
	jmp	.L96
.L95:
	movq	$37, -48(%rbp)
	jmp	.L96
.L92:
	movq	$60, -48(%rbp)
	jmp	.L96
.L93:
	movq	$62, -48(%rbp)
	nop
.L96:
	jmp	.L64
.L11:
	movq	-80(%rbp), %rax
	addq	$1, %rax
	movl	$16, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtoul@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movl	%eax, -104(%rbp)
	movq	$12, -48(%rbp)
	jmp	.L64
.L35:
	call	usage
	movq	$11, -48(%rbp)
	jmp	.L64
.L40:
	movl	$0, %eax
	jmp	.L77
.L18:
	movl	-124(%rbp), %eax
	addl	%eax, -112(%rbp)
	movq	$18, -48(%rbp)
	jmp	.L64
.L23:
	movq	-64(%rbp), %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %ecx
	movq	stderr(%rip), %rax
	movl	-112(%rbp), %edx
	leaq	.LC7(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$53, -48(%rbp)
	jmp	.L64
.L26:
	movl	$1, -124(%rbp)
	subl	$1, -132(%rbp)
	addq	$8, -144(%rbp)
	movq	$5, -48(%rbp)
	jmp	.L64
.L58:
	movq	-144(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movl	$16, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtoul@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	%eax, -112(%rbp)
	movq	-144(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movl	$45, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -80(%rbp)
	movq	$15, -48(%rbp)
	jmp	.L64
.L36:
	cmpl	$1, -132(%rbp)
	jg	.L97
	movq	$56, -48(%rbp)
	jmp	.L64
.L97:
	movq	$45, -48(%rbp)
	jmp	.L64
.L32:
	movl	-112(%rbp), %eax
	andl	$4095, %eax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -64(%rbp)
	movq	$64, -48(%rbp)
	jmp	.L64
.L7:
	cmpl	$0, -120(%rbp)
	je	.L99
	movq	$27, -48(%rbp)
	jmp	.L64
.L99:
	movq	$47, -48(%rbp)
	jmp	.L64
.L29:
	movl	-124(%rbp), %edx
	movl	-112(%rbp), %eax
	addl	%edx, %eax
	subl	$1, %eax
	movl	%eax, -104(%rbp)
	movq	$59, -48(%rbp)
	jmp	.L64
.L54:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$20, %edx
	movl	$1, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$55, -48(%rbp)
	jmp	.L64
.L28:
	movl	-112(%rbp), %eax
	andl	$4095, %eax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -56(%rbp)
	movq	$26, -48(%rbp)
	jmp	.L64
.L24:
	call	usage
	movq	$54, -48(%rbp)
	jmp	.L64
.L30:
	movq	$63, -48(%rbp)
	jmp	.L64
.L56:
	movl	$-1, %eax
	jmp	.L77
.L34:
	movl	$1, -120(%rbp)
	movq	-144(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rax
	movl	$16, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtoul@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movl	%eax, -108(%rbp)
	movq	$20, -48(%rbp)
	jmp	.L64
.L39:
	movl	$-1, %eax
	jmp	.L77
.L27:
	movl	$2, -124(%rbp)
	subl	$1, -132(%rbp)
	addq	$8, -144(%rbp)
	movq	$5, -48(%rbp)
	jmp	.L64
.L61:
	movl	-112(%rbp), %eax
	movq	%rax, -16(%rbp)
	movl	-104(%rbp), %eax
	subq	-16(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -8(%rbp)
	movq	-16(%rbp), %rcx
	movl	-116(%rbp), %edx
	movq	-8(%rbp), %rax
	movq	%rcx, %r9
	movl	%edx, %r8d
	movl	$1, %ecx
	movl	$3, %edx
	movq	%rax, %rsi
	movl	$0, %edi
	call	mmap@PLT
	movq	%rax, -88(%rbp)
	movq	$3, -48(%rbp)
	jmp	.L64
.L47:
	movl	$1052674, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -116(%rbp)
	movq	$23, -48(%rbp)
	jmp	.L64
.L102:
	nop
.L64:
	jmp	.L101
.L77:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC10:
	.string	"r [-b|-s] <address> [<value>]\n"
	.text
	.type	usage, @function
usage:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L108:
	cmpq	$0, -8(%rbp)
	je	.L104
	cmpq	$1, -8(%rbp)
	jne	.L110
	jmp	.L109
.L104:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$30, %edx
	movl	$1, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$1, -8(%rbp)
	jmp	.L107
.L110:
	nop
.L107:
	jmp	.L108
.L109:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	usage, .-usage
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
