	.file	"iijung_Code-Up-Algorithm_2020_flatten.c"
	.text
	.globl	_TIG_IZ_4cWL_argc
	.bss
	.align 4
	.type	_TIG_IZ_4cWL_argc, @object
	.size	_TIG_IZ_4cWL_argc, 4
_TIG_IZ_4cWL_argc:
	.zero	4
	.globl	_TIG_IZ_4cWL_envp
	.align 8
	.type	_TIG_IZ_4cWL_envp, @object
	.size	_TIG_IZ_4cWL_envp, 8
_TIG_IZ_4cWL_envp:
	.zero	8
	.globl	_TIG_IZ_4cWL_argv
	.align 8
	.type	_TIG_IZ_4cWL_argv, @object
	.size	_TIG_IZ_4cWL_argv, 8
_TIG_IZ_4cWL_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%s"
.LC1:
	.string	"%d"
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
	pushq	%rbx
	subq	$328, %rsp
	.cfi_offset 3, -24
	movl	%edi, -308(%rbp)
	movq	%rsi, -320(%rbp)
	movq	%rdx, -328(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_4cWL_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_4cWL_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_4cWL_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 139 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-4cWL--0
# 0 "" 2
#NO_APP
	movl	-308(%rbp), %eax
	movl	%eax, _TIG_IZ_4cWL_argc(%rip)
	movq	-320(%rbp), %rax
	movq	%rax, _TIG_IZ_4cWL_argv(%rip)
	movq	-328(%rbp), %rax
	movq	%rax, _TIG_IZ_4cWL_envp(%rip)
	nop
	movq	$12, -248(%rbp)
.L45:
	cmpq	$41, -248(%rbp)
	ja	.L48
	movq	-248(%rbp), %rax
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
	.long	.L48-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L48-.L8
	.long	.L18-.L8
	.long	.L48-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L15-.L8
	.long	.L48-.L8
	.long	.L14-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L9-.L8
	.long	.L48-.L8
	.long	.L7-.L8
	.text
.L17:
	movl	-288(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -264(%rbp)
	jge	.L31
	movq	$7, -248(%rbp)
	jmp	.L33
.L31:
	movq	$5, -248(%rbp)
	jmp	.L33
.L14:
	movl	$0, %eax
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L46
	jmp	.L47
.L27:
	movl	$0, -272(%rbp)
	movl	$0, -268(%rbp)
	movl	$0, -264(%rbp)
	movq	$18, -248(%rbp)
	jmp	.L33
.L19:
	cmpl	$19, -284(%rbp)
	jbe	.L35
	movq	$34, -248(%rbp)
	jmp	.L33
.L35:
	movq	$1, -248(%rbp)
	jmp	.L33
.L13:
	leaq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -256(%rbp)
	movq	$33, -248(%rbp)
	jmp	.L33
.L21:
	movq	$41, -248(%rbp)
	jmp	.L33
.L30:
	movl	-284(%rbp), %eax
	movl	$0, -240(%rbp,%rax,4)
	addl	$1, -284(%rbp)
	movq	$14, -248(%rbp)
	jmp	.L33
.L15:
	movl	-292(%rbp), %eax
	movb	$0, -80(%rbp,%rax)
	addl	$1, -292(%rbp)
	movq	$32, -248(%rbp)
	jmp	.L33
.L28:
	movl	-272(%rbp), %eax
	negl	%eax
	movl	%eax, -260(%rbp)
	movq	$6, -248(%rbp)
	jmp	.L33
.L18:
	movl	-264(%rbp), %eax
	cltq
	movl	-160(%rbp,%rax,4), %edx
	movl	-264(%rbp), %eax
	addl	$1, %eax
	cltq
	movl	-160(%rbp,%rax,4), %eax
	cmpl	%eax, %edx
	jge	.L37
	movq	$3, -248(%rbp)
	jmp	.L33
.L37:
	movq	$10, -248(%rbp)
	jmp	.L33
.L22:
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -288(%rbp)
	movl	$0, -240(%rbp)
	movl	$1, -284(%rbp)
	movq	$14, -248(%rbp)
	jmp	.L33
.L20:
	movl	-280(%rbp), %eax
	movl	$0, -160(%rbp,%rax,4)
	addl	$1, -280(%rbp)
	movq	$39, -248(%rbp)
	jmp	.L33
.L16:
	movl	-276(%rbp), %eax
	cltq
	movzbl	-80(%rbp,%rax), %eax
	movsbl	%al, %edx
	movl	-276(%rbp), %eax
	movl	%eax, %ecx
	shrl	$31, %ecx
	addl	%ecx, %eax
	sarl	%eax
	subl	$48, %edx
	cltq
	movl	%edx, -240(%rbp,%rax,4)
	movl	-276(%rbp), %eax
	addl	$1, %eax
	cltq
	movzbl	-80(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	-276(%rbp), %edx
	movl	%edx, %ecx
	shrl	$31, %ecx
	addl	%ecx, %edx
	sarl	%edx
	movl	%edx, %ebx
	movl	%eax, %edi
	call	getBaseVal
	movslq	%ebx, %rdx
	movl	%eax, -160(%rbp,%rdx,4)
	addl	$1, -288(%rbp)
	addl	$2, -276(%rbp)
	movq	$31, -248(%rbp)
	jmp	.L33
.L12:
	cmpl	$49, -292(%rbp)
	jbe	.L39
	movq	$11, -248(%rbp)
	jmp	.L33
.L39:
	movq	$23, -248(%rbp)
	jmp	.L33
.L25:
	movl	-260(%rbp), %eax
	addl	%eax, -268(%rbp)
	addl	$1, -264(%rbp)
	movq	$18, -248(%rbp)
	jmp	.L33
.L10:
	movl	$0, -160(%rbp)
	movl	$1, -280(%rbp)
	movq	$39, -248(%rbp)
	jmp	.L33
.L26:
	movl	-288(%rbp), %eax
	subl	$1, %eax
	cltq
	movl	-240(%rbp,%rax,4), %edx
	movl	-288(%rbp), %eax
	subl	$1, %eax
	cltq
	movl	-160(%rbp,%rax,4), %eax
	imull	%edx, %eax
	addl	%eax, -268(%rbp)
	movl	-268(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -248(%rbp)
	jmp	.L33
.L11:
	movl	-276(%rbp), %eax
	cltq
	movq	-256(%rbp), %rdx
	subq	$1, %rdx
	cmpq	%rdx, %rax
	jnb	.L41
	movq	$19, -248(%rbp)
	jmp	.L33
.L41:
	movq	$4, -248(%rbp)
	jmp	.L33
.L7:
	movb	$0, -80(%rbp)
	movl	$1, -292(%rbp)
	movq	$32, -248(%rbp)
	jmp	.L33
.L23:
	movl	-272(%rbp), %eax
	movl	%eax, -260(%rbp)
	movq	$6, -248(%rbp)
	jmp	.L33
.L9:
	cmpl	$19, -280(%rbp)
	jbe	.L43
	movq	$2, -248(%rbp)
	jmp	.L33
.L43:
	movq	$13, -248(%rbp)
	jmp	.L33
.L24:
	movl	-264(%rbp), %eax
	cltq
	movl	-240(%rbp,%rax,4), %edx
	movl	-264(%rbp), %eax
	cltq
	movl	-160(%rbp,%rax,4), %eax
	imull	%edx, %eax
	movl	%eax, -272(%rbp)
	movq	$16, -248(%rbp)
	jmp	.L33
.L29:
	movl	$0, -276(%rbp)
	movq	$31, -248(%rbp)
	jmp	.L33
.L48:
	nop
.L33:
	jmp	.L45
.L47:
	call	__stack_chk_fail@PLT
.L46:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.globl	getBaseVal
	.type	getBaseVal, @function
getBaseVal:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$7, -8(%rbp)
.L73:
	cmpq	$8, -8(%rbp)
	ja	.L74
	movq	-8(%rbp), %rax
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
	.long	.L60-.L52
	.long	.L59-.L52
	.long	.L58-.L52
	.long	.L57-.L52
	.long	.L56-.L52
	.long	.L55-.L52
	.long	.L54-.L52
	.long	.L53-.L52
	.long	.L51-.L52
	.text
.L56:
	movl	$50, %eax
	jmp	.L61
.L51:
	movl	$0, %eax
	jmp	.L61
.L59:
	movl	$5, %eax
	jmp	.L61
.L57:
	movl	$1000, %eax
	jmp	.L61
.L54:
	movl	$100, %eax
	jmp	.L61
.L55:
	movl	$10, %eax
	jmp	.L61
.L60:
	movl	$500, %eax
	jmp	.L61
.L53:
	movsbl	-20(%rbp), %eax
	subl	$67, %eax
	cmpl	$21, %eax
	ja	.L62
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L64(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L64(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L64:
	.long	.L70-.L64
	.long	.L69-.L64
	.long	.L62-.L64
	.long	.L62-.L64
	.long	.L62-.L64
	.long	.L62-.L64
	.long	.L68-.L64
	.long	.L62-.L64
	.long	.L62-.L64
	.long	.L67-.L64
	.long	.L66-.L64
	.long	.L62-.L64
	.long	.L62-.L64
	.long	.L62-.L64
	.long	.L62-.L64
	.long	.L62-.L64
	.long	.L62-.L64
	.long	.L62-.L64
	.long	.L62-.L64
	.long	.L65-.L64
	.long	.L62-.L64
	.long	.L63-.L64
	.text
.L66:
	movq	$3, -8(%rbp)
	jmp	.L71
.L69:
	movq	$0, -8(%rbp)
	jmp	.L71
.L70:
	movq	$6, -8(%rbp)
	jmp	.L71
.L67:
	movq	$4, -8(%rbp)
	jmp	.L71
.L63:
	movq	$5, -8(%rbp)
	jmp	.L71
.L65:
	movq	$1, -8(%rbp)
	jmp	.L71
.L68:
	movq	$2, -8(%rbp)
	jmp	.L71
.L62:
	movq	$8, -8(%rbp)
	nop
.L71:
	jmp	.L72
.L58:
	movl	$1, %eax
	jmp	.L61
.L74:
	nop
.L72:
	jmp	.L73
.L61:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	getBaseVal, .-getBaseVal
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
