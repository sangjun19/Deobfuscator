	.file	"ishandutta2007_codeforces-c_C_flatten.c"
	.text
	.globl	_TIG_IZ_VD28_argc
	.bss
	.align 4
	.type	_TIG_IZ_VD28_argc, @object
	.size	_TIG_IZ_VD28_argc, 4
_TIG_IZ_VD28_argc:
	.zero	4
	.globl	_TIG_IZ_VD28_argv
	.align 8
	.type	_TIG_IZ_VD28_argv, @object
	.size	_TIG_IZ_VD28_argv, 8
_TIG_IZ_VD28_argv:
	.zero	8
	.globl	_TIG_IZ_VD28_envp
	.align 8
	.type	_TIG_IZ_VD28_envp, @object
	.size	_TIG_IZ_VD28_envp, 8
_TIG_IZ_VD28_envp:
	.zero	8
	.text
	.globl	max
	.type	max, @function
max:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$3, -8(%rbp)
.L10:
	cmpq	$3, -8(%rbp)
	je	.L2
	cmpq	$3, -8(%rbp)
	ja	.L12
	cmpq	$2, -8(%rbp)
	je	.L4
	cmpq	$2, -8(%rbp)
	ja	.L12
	cmpq	$0, -8(%rbp)
	je	.L5
	cmpq	$1, -8(%rbp)
	jne	.L12
	movl	-12(%rbp), %eax
	jmp	.L11
.L2:
	movl	-20(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jle	.L7
	movq	$0, -8(%rbp)
	jmp	.L9
.L7:
	movq	$2, -8(%rbp)
	jmp	.L9
.L5:
	movl	-20(%rbp), %eax
	movl	%eax, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L9
.L4:
	movl	-24(%rbp), %eax
	movl	%eax, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L9
.L12:
	nop
.L9:
	jmp	.L10
.L11:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	max, .-max
	.section	.rodata
.LC0:
	.string	"%I64d\n"
.LC1:
	.string	"%d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-397312(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$2816, %rsp
	movl	%edi, -400100(%rbp)
	movq	%rsi, -400112(%rbp)
	movq	%rdx, -400120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_VD28_envp(%rip)
	nop
.L14:
	movq	$0, _TIG_IZ_VD28_argv(%rip)
	nop
.L15:
	movl	$0, _TIG_IZ_VD28_argc(%rip)
	nop
	nop
.L16:
.L17:
#APP
# 111 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-VD28--0
# 0 "" 2
#NO_APP
	movl	-400100(%rbp), %eax
	movl	%eax, _TIG_IZ_VD28_argc(%rip)
	movq	-400112(%rbp), %rax
	movq	%rax, _TIG_IZ_VD28_argv(%rip)
	movq	-400120(%rbp), %rax
	movq	%rax, _TIG_IZ_VD28_envp(%rip)
	nop
	movq	$8, -400056(%rbp)
.L57:
	cmpq	$38, -400056(%rbp)
	ja	.L60
	movq	-400056(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L20(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L20(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L20:
	.long	.L40-.L20
	.long	.L39-.L20
	.long	.L60-.L20
	.long	.L38-.L20
	.long	.L60-.L20
	.long	.L60-.L20
	.long	.L60-.L20
	.long	.L60-.L20
	.long	.L37-.L20
	.long	.L60-.L20
	.long	.L60-.L20
	.long	.L36-.L20
	.long	.L35-.L20
	.long	.L60-.L20
	.long	.L34-.L20
	.long	.L33-.L20
	.long	.L60-.L20
	.long	.L60-.L20
	.long	.L32-.L20
	.long	.L31-.L20
	.long	.L60-.L20
	.long	.L30-.L20
	.long	.L29-.L20
	.long	.L60-.L20
	.long	.L60-.L20
	.long	.L60-.L20
	.long	.L28-.L20
	.long	.L60-.L20
	.long	.L27-.L20
	.long	.L26-.L20
	.long	.L25-.L20
	.long	.L24-.L20
	.long	.L23-.L20
	.long	.L60-.L20
	.long	.L60-.L20
	.long	.L22-.L20
	.long	.L21-.L20
	.long	.L60-.L20
	.long	.L19-.L20
	.text
.L32:
	movl	-400088(%rbp), %eax
	cmpl	%eax, -400084(%rbp)
	jge	.L41
	movq	$22, -400056(%rbp)
	jmp	.L43
.L41:
	movq	$12, -400056(%rbp)
	jmp	.L43
.L25:
	movq	$1, -400064(%rbp)
	movl	$0, -400084(%rbp)
	movq	$18, -400056(%rbp)
	jmp	.L43
.L34:
	addl	$1, -400084(%rbp)
	movq	$32, -400056(%rbp)
	jmp	.L43
.L33:
	cmpl	$3, -400084(%rbp)
	jg	.L44
	movq	$3, -400056(%rbp)
	jmp	.L43
.L44:
	movq	$30, -400056(%rbp)
	jmp	.L43
.L24:
	movl	-400036(%rbp), %ecx
	movl	-400040(%rbp), %edx
	movl	-400044(%rbp), %esi
	movl	-400048(%rbp), %eax
	movl	%eax, %edi
	call	max4
	movl	%eax, -400068(%rbp)
	movl	-400068(%rbp), %eax
	movl	%eax, -400076(%rbp)
	movl	$0, -400072(%rbp)
	movl	$0, -400084(%rbp)
	movq	$15, -400056(%rbp)
	jmp	.L43
.L35:
	movq	-400064(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$35, -400056(%rbp)
	jmp	.L43
.L37:
	movq	$21, -400056(%rbp)
	jmp	.L43
.L39:
	cmpl	$84, -400080(%rbp)
	je	.L46
	cmpl	$84, -400080(%rbp)
	jg	.L47
	cmpl	$71, -400080(%rbp)
	je	.L48
	cmpl	$71, -400080(%rbp)
	jg	.L47
	cmpl	$65, -400080(%rbp)
	je	.L49
	cmpl	$67, -400080(%rbp)
	je	.L50
	jmp	.L47
.L46:
	movq	$29, -400056(%rbp)
	jmp	.L51
.L48:
	movq	$26, -400056(%rbp)
	jmp	.L51
.L50:
	movq	$28, -400056(%rbp)
	jmp	.L51
.L49:
	movq	$11, -400056(%rbp)
	jmp	.L51
.L47:
	movq	$14, -400056(%rbp)
	nop
.L51:
	jmp	.L43
.L38:
	movl	-400084(%rbp), %eax
	cltq
	movl	-400048(%rbp,%rax,4), %eax
	cmpl	%eax, -400076(%rbp)
	sete	%al
	movzbl	%al, %eax
	addl	%eax, -400072(%rbp)
	addl	$1, -400084(%rbp)
	movq	$15, -400056(%rbp)
	jmp	.L43
.L30:
	leaq	-400088(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -400084(%rbp)
	movq	$32, -400056(%rbp)
	jmp	.L43
.L21:
	movl	$0, -400084(%rbp)
	movq	$38, -400056(%rbp)
	jmp	.L43
.L28:
	movl	-400084(%rbp), %eax
	cltq
	movl	$2, -400032(%rbp,%rax,4)
	movq	$14, -400056(%rbp)
	jmp	.L43
.L36:
	movl	-400084(%rbp), %eax
	cltq
	movl	$0, -400032(%rbp,%rax,4)
	movq	$14, -400056(%rbp)
	jmp	.L43
.L31:
	call	getchar@PLT
	movl	%eax, -400080(%rbp)
	movq	$1, -400056(%rbp)
	jmp	.L43
.L23:
	movl	-400088(%rbp), %eax
	cmpl	%eax, -400084(%rbp)
	jge	.L52
	movq	$19, -400056(%rbp)
	jmp	.L43
.L52:
	movq	$36, -400056(%rbp)
	jmp	.L43
.L19:
	movl	-400088(%rbp), %eax
	cmpl	%eax, -400084(%rbp)
	jge	.L54
	movq	$0, -400056(%rbp)
	jmp	.L43
.L54:
	movq	$31, -400056(%rbp)
	jmp	.L43
.L29:
	movl	-400072(%rbp), %eax
	cltq
	imulq	-400064(%rbp), %rax
	movq	%rax, %rcx
	movabsq	$-8543223828751151131, %rdx
	movq	%rcx, %rax
	imulq	%rdx
	leaq	(%rdx,%rcx), %rax
	sarq	$29, %rax
	movq	%rcx, %rdx
	sarq	$63, %rdx
	subq	%rdx, %rax
	movq	%rax, -400064(%rbp)
	movq	-400064(%rbp), %rax
	imulq	$1000000007, %rax, %rdx
	movq	%rcx, %rax
	subq	%rdx, %rax
	movq	%rax, -400064(%rbp)
	addl	$1, -400084(%rbp)
	movq	$18, -400056(%rbp)
	jmp	.L43
.L27:
	movl	-400084(%rbp), %eax
	cltq
	movl	$1, -400032(%rbp,%rax,4)
	movq	$14, -400056(%rbp)
	jmp	.L43
.L40:
	movl	-400084(%rbp), %eax
	cltq
	movl	-400032(%rbp,%rax,4), %edx
	movl	%edx, %eax
	movl	-400048(%rbp,%rax,4), %eax
	addl	$1, %eax
	movl	%edx, %edx
	movl	%eax, -400048(%rbp,%rdx,4)
	addl	$1, -400084(%rbp)
	movq	$38, -400056(%rbp)
	jmp	.L43
.L22:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L58
	jmp	.L59
.L26:
	movl	-400084(%rbp), %eax
	cltq
	movl	$3, -400032(%rbp,%rax,4)
	movq	$14, -400056(%rbp)
	jmp	.L43
.L60:
	nop
.L43:
	jmp	.L57
.L59:
	call	__stack_chk_fail@PLT
.L58:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.globl	max4
	.type	max4, @function
max4:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movl	%esi, -40(%rbp)
	movl	%edx, -44(%rbp)
	movl	%ecx, -48(%rbp)
	movq	$0, -8(%rbp)
.L67:
	cmpq	$2, -8(%rbp)
	je	.L62
	cmpq	$2, -8(%rbp)
	ja	.L69
	cmpq	$0, -8(%rbp)
	je	.L64
	cmpq	$1, -8(%rbp)
	jne	.L69
	movl	-48(%rbp), %edx
	movl	-44(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	max
	movl	%eax, -16(%rbp)
	movl	-40(%rbp), %edx
	movl	-36(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	max
	movl	%eax, -12(%rbp)
	movl	-16(%rbp), %edx
	movl	-12(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	max
	movl	%eax, -20(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L65
.L64:
	movq	$1, -8(%rbp)
	jmp	.L65
.L62:
	movl	-20(%rbp), %eax
	jmp	.L68
.L69:
	nop
.L65:
	jmp	.L67
.L68:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	max4, .-max4
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
