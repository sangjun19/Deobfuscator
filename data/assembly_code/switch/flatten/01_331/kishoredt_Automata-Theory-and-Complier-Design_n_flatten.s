	.file	"kishoredt_Automata-Theory-and-Complier-Design_n_flatten.c"
	.text
	.globl	_TIG_IZ_6B0n_envp
	.bss
	.align 8
	.type	_TIG_IZ_6B0n_envp, @object
	.size	_TIG_IZ_6B0n_envp, 8
_TIG_IZ_6B0n_envp:
	.zero	8
	.globl	_TIG_IZ_6B0n_argv
	.align 8
	.type	_TIG_IZ_6B0n_argv, @object
	.size	_TIG_IZ_6B0n_argv, 8
_TIG_IZ_6B0n_argv:
	.zero	8
	.globl	_TIG_IZ_6B0n_argc
	.align 4
	.type	_TIG_IZ_6B0n_argc, @object
	.size	_TIG_IZ_6B0n_argc, 4
_TIG_IZ_6B0n_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"enter the string to be checked: "
.LC1:
	.string	"%s"
.LC2:
	.string	"String is not accepted"
.LC3:
	.string	"String is accepted"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_6B0n_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_6B0n_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_6B0n_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-6B0n--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_6B0n_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_6B0n_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_6B0n_envp(%rip)
	nop
	movq	$20, -120(%rbp)
.L43:
	cmpq	$20, -120(%rbp)
	ja	.L46
	movq	-120(%rbp), %rax
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
	.long	.L25-.L8
	.long	.L46-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L46-.L8
	.long	.L46-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	-124(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	cmpb	$48, %al
	jne	.L26
	movq	$19, -120(%rbp)
	jmp	.L28
.L26:
	movq	$0, -120(%rbp)
	jmp	.L28
.L22:
	movl	-124(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	cmpb	$49, %al
	jne	.L29
	movq	$10, -120(%rbp)
	jmp	.L28
.L29:
	movq	$18, -120(%rbp)
	jmp	.L28
.L13:
	movl	-124(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	cmpb	$48, %al
	jne	.L31
	movq	$5, -120(%rbp)
	jmp	.L28
.L31:
	movq	$4, -120(%rbp)
	jmp	.L28
.L14:
	movsbl	-125(%rbp), %eax
	cmpl	$97, %eax
	je	.L33
	cmpl	$98, %eax
	jne	.L34
	movq	$15, -120(%rbp)
	jmp	.L35
.L33:
	movq	$11, -120(%rbp)
	jmp	.L35
.L34:
	movq	$9, -120(%rbp)
	nop
.L35:
	jmp	.L28
.L18:
	movb	$97, -125(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -124(%rbp)
	movq	$3, -120(%rbp)
	jmp	.L28
.L23:
	movl	-124(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	testb	%al, %al
	je	.L36
	movq	$12, -120(%rbp)
	jmp	.L28
.L36:
	movq	$6, -120(%rbp)
	jmp	.L28
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L44
	jmp	.L45
.L15:
	movl	-124(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	cmpb	$49, %al
	jne	.L39
	movq	$7, -120(%rbp)
	jmp	.L28
.L39:
	movq	$0, -120(%rbp)
	jmp	.L28
.L17:
	movq	$0, -120(%rbp)
	jmp	.L28
.L9:
	movb	$99, -125(%rbp)
	movq	$0, -120(%rbp)
	jmp	.L28
.L11:
	movsbl	-125(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -120(%rbp)
	jmp	.L28
.L20:
	cmpb	$97, -125(%rbp)
	jne	.L41
	movq	$2, -120(%rbp)
	jmp	.L28
.L41:
	movq	$17, -120(%rbp)
	jmp	.L28
.L21:
	movb	$97, -125(%rbp)
	movq	$0, -120(%rbp)
	jmp	.L28
.L16:
	movb	$99, -125(%rbp)
	movq	$0, -120(%rbp)
	jmp	.L28
.L25:
	addl	$1, -124(%rbp)
	movq	$3, -120(%rbp)
	jmp	.L28
.L19:
	movb	$98, -125(%rbp)
	movq	$0, -120(%rbp)
	jmp	.L28
.L24:
	movsbl	-125(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -120(%rbp)
	jmp	.L28
.L7:
	movq	$8, -120(%rbp)
	jmp	.L28
.L46:
	nop
.L28:
	jmp	.L43
.L45:
	call	__stack_chk_fail@PLT
.L44:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
