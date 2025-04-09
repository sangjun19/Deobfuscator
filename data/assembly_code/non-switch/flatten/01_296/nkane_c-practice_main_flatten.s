	.file	"nkane_c-practice_main_flatten.c"
	.text
	.globl	_TIG_IZ_gwVR_argv
	.bss
	.align 8
	.type	_TIG_IZ_gwVR_argv, @object
	.size	_TIG_IZ_gwVR_argv, 8
_TIG_IZ_gwVR_argv:
	.zero	8
	.globl	_TIG_IZ_gwVR_argc
	.align 4
	.type	_TIG_IZ_gwVR_argc, @object
	.size	_TIG_IZ_gwVR_argc, 4
_TIG_IZ_gwVR_argc:
	.zero	4
	.globl	_TIG_IZ_gwVR_envp
	.align 8
	.type	_TIG_IZ_gwVR_envp, @object
	.size	_TIG_IZ_gwVR_envp, 8
_TIG_IZ_gwVR_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"\nThe file %s has successfully been written as a binary file.\n"
	.align 8
.LC1:
	.string	"\nThe existing file %s will not be overwritten.\n"
.LC3:
	.string	"r"
	.align 8
.LC4:
	.string	"\nA file by the name %s exists."
	.align 8
.LC5:
	.string	"\nDo you want to continue and overwrite it"
.LC6:
	.string	"\nwith the new data (y or n): "
.LC7:
	.string	"%c"
.LC8:
	.string	"wb"
	.align 8
.LC9:
	.string	"\nThe file %s was not successfully opened.\n"
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
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_gwVR_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_gwVR_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_gwVR_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-gwVR--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_gwVR_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_gwVR_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_gwVR_envp(%rip)
	nop
	movq	$5, -40(%rbp)
.L32:
	cmpq	$21, -40(%rbp)
	ja	.L35
	movq	-40(%rbp), %rax
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
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L35-.L8
	.long	.L15-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L14-.L8
	.long	.L35-.L8
	.long	.L13-.L8
	.long	.L35-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L35-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L17:
	movl	-68(%rbp), %eax
	movb	$0, -32(%rbp,%rax)
	addl	$1, -68(%rbp)
	movq	$3, -40(%rbp)
	jmp	.L22
.L13:
	movq	-48(%rbp), %rdx
	leaq	-72(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-48(%rbp), %rdx
	leaq	-64(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$8, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-48(%rbp), %rdx
	leaq	-56(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$8, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$17, -40(%rbp)
	jmp	.L22
.L14:
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	$1, %edi
	call	exit@PLT
.L20:
	movzbl	-73(%rbp), %eax
	cmpb	$110, %al
	jne	.L23
	movq	$12, -40(%rbp)
	jmp	.L22
.L23:
	movq	$2, -40(%rbp)
	jmp	.L22
.L18:
	cmpl	$19, -68(%rbp)
	jbe	.L25
	movq	$21, -40(%rbp)
	jmp	.L22
.L25:
	movq	$4, -40(%rbp)
	jmp	.L22
.L12:
	cmpq	$0, -48(%rbp)
	je	.L27
	movq	$0, -40(%rbp)
	jmp	.L22
.L27:
	movq	$2, -40(%rbp)
	jmp	.L22
.L7:
	movl	$125, -72(%rbp)
	movq	$-125, -64(%rbp)
	movsd	.LC2(%rip), %xmm0
	movsd	%xmm0, -56(%rbp)
	leaq	-32(%rbp), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -48(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L22
.L10:
	movb	$46, -32(%rbp)
	movb	$47, -31(%rbp)
	movb	$100, -30(%rbp)
	movb	$97, -29(%rbp)
	movb	$116, -28(%rbp)
	movb	$97, -27(%rbp)
	movb	$47, -26(%rbp)
	movb	$112, -25(%rbp)
	movb	$114, -24(%rbp)
	movb	$105, -23(%rbp)
	movb	$99, -22(%rbp)
	movb	$101, -21(%rbp)
	movb	$115, -20(%rbp)
	movb	$46, -19(%rbp)
	movb	$98, -18(%rbp)
	movb	$105, -17(%rbp)
	movb	$110, -16(%rbp)
	movb	$0, -15(%rbp)
	movl	$18, -68(%rbp)
	movq	$3, -40(%rbp)
	jmp	.L22
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	jmp	.L34
.L16:
	movq	$19, -40(%rbp)
	jmp	.L22
.L21:
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-73(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -40(%rbp)
	jmp	.L22
.L15:
	cmpq	$0, -48(%rbp)
	jne	.L30
	movq	$20, -40(%rbp)
	jmp	.L22
.L30:
	movq	$14, -40(%rbp)
	jmp	.L22
.L19:
	leaq	-32(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -48(%rbp)
	movq	$7, -40(%rbp)
	jmp	.L22
.L9:
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	exit@PLT
.L35:
	nop
.L22:
	jmp	.L32
.L34:
	call	__stack_chk_fail@PLT
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC2:
	.long	343597384
	.long	1072777134
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
