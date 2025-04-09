	.file	"Melearner777_Guvi-Ds-with-C_8_flatten.c"
	.text
	.globl	_TIG_IZ_rW9d_envp
	.bss
	.align 8
	.type	_TIG_IZ_rW9d_envp, @object
	.size	_TIG_IZ_rW9d_envp, 8
_TIG_IZ_rW9d_envp:
	.zero	8
	.globl	_TIG_IZ_rW9d_argc
	.align 4
	.type	_TIG_IZ_rW9d_argc, @object
	.size	_TIG_IZ_rW9d_argc, 4
_TIG_IZ_rW9d_argc:
	.zero	4
	.globl	_TIG_IZ_rW9d_argv
	.align 8
	.type	_TIG_IZ_rW9d_argv, @object
	.size	_TIG_IZ_rW9d_argv, 8
_TIG_IZ_rW9d_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"Days in reverse order:"
.LC2:
	.string	"%s"
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_rW9d_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_rW9d_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_rW9d_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-rW9d--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_rW9d_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_rW9d_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_rW9d_envp(%rip)
	nop
	movq	$8, -32(%rbp)
.L33:
	cmpq	$21, -32(%rbp)
	ja	.L36
	movq	-32(%rbp), %rax
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
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L36-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L36-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L36-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L18:
	cmpl	$0, -44(%rbp)
	js	.L22
	movq	$0, -32(%rbp)
	jmp	.L24
.L22:
	movq	$15, -32(%rbp)
	jmp	.L24
.L12:
	movl	$10, %edi
	call	putchar@PLT
	movq	$20, -32(%rbp)
	jmp	.L24
.L15:
	movq	$7, -32(%rbp)
	jmp	.L24
.L20:
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
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
.L25:
	cmpq	%rdx, %rsp
	je	.L26
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L25
.L26:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L27
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L27:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -40(%rbp)
	movq	$21, -32(%rbp)
	jmp	.L24
.L11:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -48(%rbp)
	movq	$5, -32(%rbp)
	jmp	.L24
.L7:
	movl	$0, -48(%rbp)
	movq	$5, -32(%rbp)
	jmp	.L24
.L13:
	cmpl	$0, -44(%rbp)
	jle	.L28
	movq	$2, -32(%rbp)
	jmp	.L24
.L28:
	movq	$17, -32(%rbp)
	jmp	.L24
.L10:
	subl	$1, -44(%rbp)
	movq	$4, -32(%rbp)
	jmp	.L24
.L17:
	movl	-52(%rbp), %eax
	cmpl	%eax, -48(%rbp)
	jge	.L30
	movq	$16, -32(%rbp)
	jmp	.L24
.L30:
	movq	$10, -32(%rbp)
	jmp	.L24
.L14:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-52(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -44(%rbp)
	movq	$4, -32(%rbp)
	jmp	.L24
.L21:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	getDayOfWeekString
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -32(%rbp)
	jmp	.L24
.L16:
	leaq	-52(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -32(%rbp)
	jmp	.L24
.L19:
	movl	$32, %edi
	call	putchar@PLT
	movq	$17, -32(%rbp)
	jmp	.L24
.L9:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L34
	jmp	.L35
.L36:
	nop
.L24:
	jmp	.L33
.L35:
	call	__stack_chk_fail@PLT
.L34:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.rodata
.LC3:
	.string	"Thursday"
.LC4:
	.string	"Invalid"
.LC5:
	.string	"Friday"
.LC6:
	.string	"Wednesday"
.LC7:
	.string	"Tuesday"
.LC8:
	.string	"Saturday"
.LC9:
	.string	"Sunday"
.LC10:
	.string	"Monday"
	.text
	.globl	getDayOfWeekString
	.type	getDayOfWeekString, @function
getDayOfWeekString:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$5, -8(%rbp)
.L61:
	cmpq	$9, -8(%rbp)
	ja	.L62
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L40(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L40(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L40:
	.long	.L48-.L40
	.long	.L47-.L40
	.long	.L46-.L40
	.long	.L45-.L40
	.long	.L62-.L40
	.long	.L44-.L40
	.long	.L43-.L40
	.long	.L42-.L40
	.long	.L41-.L40
	.long	.L39-.L40
	.text
.L41:
	leaq	.LC3(%rip), %rax
	jmp	.L49
.L47:
	leaq	.LC4(%rip), %rax
	jmp	.L49
.L45:
	leaq	.LC5(%rip), %rax
	jmp	.L49
.L39:
	leaq	.LC6(%rip), %rax
	jmp	.L49
.L43:
	leaq	.LC7(%rip), %rax
	jmp	.L49
.L44:
	cmpl	$6, -20(%rbp)
	ja	.L50
	movl	-20(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L52(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L52(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L52:
	.long	.L58-.L52
	.long	.L57-.L52
	.long	.L56-.L52
	.long	.L55-.L52
	.long	.L54-.L52
	.long	.L53-.L52
	.long	.L51-.L52
	.text
.L51:
	movq	$0, -8(%rbp)
	jmp	.L59
.L53:
	movq	$3, -8(%rbp)
	jmp	.L59
.L54:
	movq	$8, -8(%rbp)
	jmp	.L59
.L55:
	movq	$9, -8(%rbp)
	jmp	.L59
.L56:
	movq	$6, -8(%rbp)
	jmp	.L59
.L57:
	movq	$2, -8(%rbp)
	jmp	.L59
.L58:
	movq	$7, -8(%rbp)
	jmp	.L59
.L50:
	movq	$1, -8(%rbp)
	nop
.L59:
	jmp	.L60
.L48:
	leaq	.LC8(%rip), %rax
	jmp	.L49
.L42:
	leaq	.LC9(%rip), %rax
	jmp	.L49
.L46:
	leaq	.LC10(%rip), %rax
	jmp	.L49
.L62:
	nop
.L60:
	jmp	.L61
.L49:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	getDayOfWeekString, .-getDayOfWeekString
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
