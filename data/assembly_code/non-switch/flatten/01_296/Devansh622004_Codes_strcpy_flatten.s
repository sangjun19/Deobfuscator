	.file	"Devansh622004_Codes_strcpy_flatten.c"
	.text
	.globl	_TIG_IZ_jbch_envp
	.bss
	.align 8
	.type	_TIG_IZ_jbch_envp, @object
	.size	_TIG_IZ_jbch_envp, 8
_TIG_IZ_jbch_envp:
	.zero	8
	.globl	_TIG_IZ_jbch_argc
	.align 4
	.type	_TIG_IZ_jbch_argc, @object
	.size	_TIG_IZ_jbch_argc, 4
_TIG_IZ_jbch_argc:
	.zero	4
	.globl	_TIG_IZ_jbch_argv
	.align 8
	.type	_TIG_IZ_jbch_argv, @object
	.size	_TIG_IZ_jbch_argv, 8
_TIG_IZ_jbch_argv:
	.zero	8
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
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_jbch_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_jbch_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_jbch_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 129 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-jbch--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_jbch_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_jbch_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_jbch_envp(%rip)
	nop
	movq	$6, -80(%rbp)
.L22:
	cmpq	$11, -80(%rbp)
	ja	.L25
	movq	-80(%rbp), %rax
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
	.long	.L15-.L8
	.long	.L25-.L8
	.long	.L26-.L8
	.long	.L13-.L8
	.long	.L25-.L8
	.long	.L25-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L25-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	cmpl	$19, -92(%rbp)
	jbe	.L16
	movq	$3, -80(%rbp)
	jmp	.L18
.L16:
	movq	$7, -80(%rbp)
	jmp	.L18
.L13:
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
	movl	%eax, -84(%rbp)
	movl	$0, -88(%rbp)
	movq	$11, -80(%rbp)
	jmp	.L18
.L7:
	movl	-88(%rbp), %eax
	cmpl	-84(%rbp), %eax
	jge	.L19
	movq	$10, -80(%rbp)
	jmp	.L18
.L19:
	movq	$2, -80(%rbp)
	jmp	.L18
.L12:
	movq	$0, -80(%rbp)
	jmp	.L18
.L9:
	movl	-88(%rbp), %eax
	cltq
	movzbl	-64(%rbp,%rax), %edx
	movl	-88(%rbp), %eax
	cltq
	movb	%dl, -32(%rbp,%rax)
	movl	-88(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$1, -88(%rbp)
	movq	$11, -80(%rbp)
	jmp	.L18
.L15:
	movb	$106, -64(%rbp)
	movb	$97, -63(%rbp)
	movb	$105, -62(%rbp)
	movb	$32, -61(%rbp)
	movb	$115, -60(%rbp)
	movb	$104, -59(%rbp)
	movb	$114, -58(%rbp)
	movb	$101, -57(%rbp)
	movb	$101, -56(%rbp)
	movb	$32, -55(%rbp)
	movb	$114, -54(%rbp)
	movb	$97, -53(%rbp)
	movb	$109, -52(%rbp)
	movb	$0, -51(%rbp)
	movl	$14, -92(%rbp)
	movq	$8, -80(%rbp)
	jmp	.L18
.L11:
	movl	-92(%rbp), %eax
	movb	$0, -64(%rbp,%rax)
	addl	$1, -92(%rbp)
	movq	$8, -80(%rbp)
	jmp	.L18
.L25:
	nop
.L18:
	jmp	.L22
.L26:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L24
	call	__stack_chk_fail@PLT
.L24:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
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
