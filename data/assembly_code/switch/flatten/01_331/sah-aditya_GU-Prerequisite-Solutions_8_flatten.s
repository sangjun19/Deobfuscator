	.file	"sah-aditya_GU-Prerequisite-Solutions_8_flatten.c"
	.text
	.globl	_TIG_IZ_jCNg_envp
	.bss
	.align 8
	.type	_TIG_IZ_jCNg_envp, @object
	.size	_TIG_IZ_jCNg_envp, 8
_TIG_IZ_jCNg_envp:
	.zero	8
	.globl	_TIG_IZ_jCNg_argc
	.align 4
	.type	_TIG_IZ_jCNg_argc, @object
	.size	_TIG_IZ_jCNg_argc, 4
_TIG_IZ_jCNg_argc:
	.zero	4
	.globl	_TIG_IZ_jCNg_argv
	.align 8
	.type	_TIG_IZ_jCNg_argv, @object
	.size	_TIG_IZ_jCNg_argv, 8
_TIG_IZ_jCNg_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Days in reverse order:"
.LC1:
	.string	"%d"
.LC2:
	.string	"Friday"
.LC3:
	.string	"Tuesday"
.LC4:
	.string	"Sunday"
.LC5:
	.string	"Wednesday"
.LC6:
	.string	"Monday"
.LC7:
	.string	"Saturday"
.LC8:
	.string	"Thursday"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_jCNg_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_jCNg_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_jCNg_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 90 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-jCNg--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_jCNg_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_jCNg_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_jCNg_envp(%rip)
	nop
	movq	$17, -24(%rbp)
.L51:
	cmpq	$35, -24(%rbp)
	ja	.L54
	movq	-24(%rbp), %rax
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
	.long	.L29-.L8
	.long	.L54-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L22-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L54-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L54-.L8
	.long	.L17-.L8
	.long	.L54-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L12-.L8
	.long	.L54-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L15:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-48(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -36(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L30
.L26:
	cmpl	$0, -36(%rbp)
	js	.L31
	movq	$30, -24(%rbp)
	jmp	.L30
.L31:
	movq	$22, -24(%rbp)
	jmp	.L30
.L12:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$6, %eax
	ja	.L33
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L35(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L35(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L35:
	.long	.L41-.L35
	.long	.L40-.L35
	.long	.L39-.L35
	.long	.L38-.L35
	.long	.L37-.L35
	.long	.L36-.L35
	.long	.L34-.L35
	.text
.L34:
	movq	$10, -24(%rbp)
	jmp	.L42
.L36:
	movq	$16, -24(%rbp)
	jmp	.L42
.L37:
	movq	$2, -24(%rbp)
	jmp	.L42
.L38:
	movq	$32, -24(%rbp)
	jmp	.L42
.L39:
	movq	$26, -24(%rbp)
	jmp	.L42
.L40:
	movq	$27, -24(%rbp)
	jmp	.L42
.L41:
	movq	$19, -24(%rbp)
	jmp	.L42
.L33:
	movq	$5, -24(%rbp)
	nop
.L42:
	jmp	.L30
.L27:
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-44(%rbp), %edx
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-32(%rbp), %rax
	addq	%rcx, %rax
	movl	%edx, (%rax)
	addl	$1, -40(%rbp)
	movq	$0, -24(%rbp)
	jmp	.L30
.L21:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$34, -24(%rbp)
	jmp	.L30
.L16:
	movl	$32, %edi
	call	putchar@PLT
	movq	$7, -24(%rbp)
	jmp	.L30
.L14:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$34, -24(%rbp)
	jmp	.L30
.L19:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$34, -24(%rbp)
	jmp	.L30
.L11:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$34, -24(%rbp)
	jmp	.L30
.L20:
	movq	$33, -24(%rbp)
	jmp	.L30
.L24:
	movl	$0, -40(%rbp)
	movq	$0, -24(%rbp)
	jmp	.L30
.L13:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$34, -24(%rbp)
	jmp	.L30
.L9:
	cmpl	$0, -36(%rbp)
	jle	.L43
	movq	$24, -24(%rbp)
	jmp	.L30
.L43:
	movq	$7, -24(%rbp)
	jmp	.L30
.L17:
	movl	$10, %edi
	call	putchar@PLT
	movq	$20, -24(%rbp)
	jmp	.L30
.L25:
	movq	$34, -24(%rbp)
	jmp	.L30
.L10:
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$35, -24(%rbp)
	jmp	.L30
.L22:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$34, -24(%rbp)
	jmp	.L30
.L29:
	movl	-48(%rbp), %eax
	cmpl	%eax, -40(%rbp)
	jge	.L45
	movq	$3, -24(%rbp)
	jmp	.L30
.L45:
	movq	$25, -24(%rbp)
	jmp	.L30
.L23:
	subl	$1, -36(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L30
.L7:
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %esi
	movl	$0, %edx
	divq	%rsi
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L47:
	cmpq	%rdx, %rsp
	je	.L48
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L47
.L48:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L49
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L49:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -32(%rbp)
	movq	$6, -24(%rbp)
	jmp	.L30
.L28:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$34, -24(%rbp)
	jmp	.L30
.L18:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L52
	jmp	.L53
.L54:
	nop
.L30:
	jmp	.L51
.L53:
	call	__stack_chk_fail@PLT
.L52:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
