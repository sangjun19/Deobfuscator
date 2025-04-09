	.file	"mmk-manavmistry_Tops-Data_pr_6_flatten.c"
	.text
	.globl	_TIG_IZ_g5Hm_argc
	.bss
	.align 4
	.type	_TIG_IZ_g5Hm_argc, @object
	.size	_TIG_IZ_g5Hm_argc, 4
_TIG_IZ_g5Hm_argc:
	.zero	4
	.globl	_TIG_IZ_g5Hm_argv
	.align 8
	.type	_TIG_IZ_g5Hm_argv, @object
	.size	_TIG_IZ_g5Hm_argv, 8
_TIG_IZ_g5Hm_argv:
	.zero	8
	.globl	_TIG_IZ_g5Hm_envp
	.align 8
	.type	_TIG_IZ_g5Hm_envp, @object
	.size	_TIG_IZ_g5Hm_envp, 8
_TIG_IZ_g5Hm_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Enter a string: "
.LC1:
	.string	"%[^\n]s"
.LC2:
	.string	"Total vowels: %d\n"
.LC3:
	.string	"Total consonants: %d\n"
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
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_g5Hm_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_g5Hm_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_g5Hm_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 129 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-g5Hm--0
# 0 "" 2
#NO_APP
	movl	-148(%rbp), %eax
	movl	%eax, _TIG_IZ_g5Hm_argc(%rip)
	movq	-160(%rbp), %rax
	movq	%rax, _TIG_IZ_g5Hm_argv(%rip)
	movq	-168(%rbp), %rax
	movq	%rax, _TIG_IZ_g5Hm_envp(%rip)
	nop
	movq	$5, -120(%rbp)
.L30:
	cmpq	$17, -120(%rbp)
	ja	.L33
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
	.long	.L33-.L8
	.long	.L19-.L8
	.long	.L33-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L33-.L8
	.long	.L10-.L8
	.long	.L33-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L17:
	movl	-128(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	tolower@PLT
	movl	%eax, -124(%rbp)
	movl	-124(%rbp), %eax
	movb	%al, -137(%rbp)
	movq	$11, -120(%rbp)
	jmp	.L20
.L10:
	movsbl	-137(%rbp), %eax
	subl	$97, %eax
	cmpl	$20, %eax
	seta	%dl
	testb	%dl, %dl
	jne	.L21
	movl	$1065233, %edx
	movl	%eax, %ecx
	shrq	%cl, %rdx
	movq	%rdx, %rax
	andl	$1, %eax
	testq	%rax, %rax
	setne	%al
	testb	%al, %al
	je	.L21
	movq	$6, -120(%rbp)
	jmp	.L22
.L21:
	movq	$1, -120(%rbp)
	nop
.L22:
	jmp	.L20
.L11:
	addl	$1, -128(%rbp)
	movq	$3, -120(%rbp)
	jmp	.L20
.L19:
	addl	$1, -132(%rbp)
	movq	$12, -120(%rbp)
	jmp	.L20
.L18:
	movl	-128(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	testb	%al, %al
	je	.L23
	movq	$4, -120(%rbp)
	jmp	.L20
.L23:
	movq	$7, -120(%rbp)
	jmp	.L20
.L9:
	movl	$0, -136(%rbp)
	movl	$0, -132(%rbp)
	movl	$0, -128(%rbp)
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
	movq	$3, -120(%rbp)
	jmp	.L20
.L12:
	cmpb	$96, -137(%rbp)
	jle	.L25
	movq	$17, -120(%rbp)
	jmp	.L20
.L25:
	movq	$12, -120(%rbp)
	jmp	.L20
.L7:
	cmpb	$122, -137(%rbp)
	jg	.L27
	movq	$14, -120(%rbp)
	jmp	.L20
.L27:
	movq	$12, -120(%rbp)
	jmp	.L20
.L15:
	addl	$1, -136(%rbp)
	movq	$12, -120(%rbp)
	jmp	.L20
.L16:
	movq	$16, -120(%rbp)
	jmp	.L20
.L13:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L31
	jmp	.L32
.L14:
	movl	-136(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-132(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -120(%rbp)
	jmp	.L20
.L33:
	nop
.L20:
	jmp	.L30
.L32:
	call	__stack_chk_fail@PLT
.L31:
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
